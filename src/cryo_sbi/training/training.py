import os
import time
import logging
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from itertools import islice
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cryo_sbi.simulator.priors import PriorLoader
from cryo_sbi.simulator.cryo_em_simulator import CryoEmSimulator
from cryo_sbi.models.build_models import build_classifier


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class ClassifierLoss(nn.Module):
    def __init__(self, estimator: nn.Module, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.estimator = estimator
        self.label_smoothing = label_smoothing

    def forward(
        self, indices: torch.Tensor, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.estimator(images)
        loss = nn.functional.cross_entropy(
            logits, indices, reduction="mean", label_smoothing=self.label_smoothing
        )
        return loss, logits


class GDStep:
    """
    One optimizer step, optionally with gradient clipping, LR scheduling, and
    AMP gradient scaling.

    The scaler is always used — when disabled (e.g. fp32 training or CPU), every
    GradScaler call is a no-op, so the same code path handles both modes.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        clip: float = None,
        lr_scheduler=None,
        scaler=None,
    ) -> None:
        self.optimizer = optimizer
        self.parameters = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        self.clip = clip
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=False)

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        if not loss.isfinite().all():
            return loss.detach(), None

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        if self.clip is not None:
            # unscale before clipping so the threshold has its real meaning.
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
        else:
            grad_norm = None

        # scaler.step internally skips optimizer.step() if grads are non-finite.
        # Compare the loss scale before vs. after update() to detect a skip:
        # scale shrinks on a skipped step, stays same or grows on a successful one.
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        stepped = self.scaler.get_scale() >= scale_before

        # Only step the scheduler when the optimizer actually stepped — otherwise
        # OneCycleLR drifts off its schedule after a single non-finite grad batch.
        if stepped and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.detach(), grad_norm


def _underlying_module(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module, unwrapping torch.compile if present."""
    return getattr(model, "_orig_mod", model)


def _save_checkpoint(
    path: str,
    estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    epoch: int,
    scaler=None,
) -> None:
    """Save a full training checkpoint (model + optimizer + scheduler + epoch + RNG)."""
    state = {
        "model": _underlying_module(estimator).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": int(epoch),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(state, path)


def _load_checkpoint(
    path: str,
    estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler=None,
) -> int:
    """
    Load a training checkpoint into the given model/optimizer/scheduler.

    Accepts both the new dict format (model + optimizer + scheduler + epoch + RNG)
    and the legacy state-dict-only format. For legacy checkpoints, only model
    weights are restored and the resumed epoch is 0.

    Returns:
        int: epoch index to resume from (0 for legacy checkpoints).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, weights_only=False, map_location="cpu")

    # Legacy: a raw state_dict — keys are weight tensor names.
    if not isinstance(state, dict) or "model" not in state:
        logging.warning(
            f"Loading legacy weights-only checkpoint from {path}; "
            "optimizer / scheduler / epoch / RNG state will not be restored."
        )
        _underlying_module(estimator).load_state_dict(state)
        return 0

    _underlying_module(estimator).load_state_dict(state["model"])
    if "optimizer" in state and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and state.get("scheduler") is not None:
        lr_scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if state.get("rng_state") is not None:
        torch.set_rng_state(state["rng_state"].cpu().to(torch.uint8))
    if (
        state.get("cuda_rng_state") is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    start_epoch = int(state.get("epoch", 0))
    logging.info(f"Resuming from {path} at epoch {start_epoch}")
    return start_epoch

# training diagnostic functions
def labels_from_parameters(parameters, simulator):
    """Convert simulator parameters into classifier labels."""
    fg_indices = parameters[0]

    if fg_indices.ndim == 2:
        labels = fg_indices[:, 0].long().clone()
    else:
        labels = fg_indices.long().clone()
    
    if simulator.garbage_class:
        garbage_mask = parameters[13].bool()
        labels[garbage_mask] = simulator.num_models

    return labels

@torch.no_grad()
def evaluate_probe_set(
    estimator,
    probe_images,
    probe_labels,
    device,
    num_classes,
    batch_size,
):
    """Evaluate the current model on the fixed diagnostic images."""
    was_training = estimator.training
    estimator.eval()

    logits_list = []

    for start in range(0, len(probe_images), batch_size):
        end = start + batch_size

        images = probe_images[start:end].to(
            device,
            non_blocking=True,
        )

        logits = estimator(images)
        logits_list.append(logits.float().cpu())
    
    if was_training:
        estimator.train()

    logits = torch.cat(logits_list, dim=0)
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)

    # Keep one loss value per image
    image_losses = F.cross_entropy(
        logits,
        probe_labels,
        reduction="none",
    )

    correct = predictions.eq(probe_labels)

    class_counts = torch.bincount(
        probe_labels,
        minlength=num_classes,
    ).float()

    class_correct = torch.bincount(
        probe_labels,
        weights=correct.float(),
        minlength=num_classes,
    )

    class_loss_sum = torch.bincount(
        probe_labels,
        weights=image_losses,
        minlength=num_classes,
    )

    valid_classes = class_counts > 0

    class_accuracy = torch.full(
        (num_classes,),
        float("nan"),
    )

    class_loss = torch.full(
        (num_classes,),
        float("nan"),
    )

    class_accuracy[valid_classes] = (
        class_correct[valid_classes]
        / class_counts[valid_classes]
    )

    class_loss[valid_classes] = (
        class_loss_sum[valid_classes]
        / class_counts[valid_classes]
    )

    macro_accuracy = class_accuracy[valid_classes].mean()

    true_class_probability = probabilities.gather(
        1,
        probe_labels[:, None],
    ).squeeze(1)

    predicted_probability = probabilities.max(dim=1).values

    flat_confusion_indices = (
        probe_labels * num_classes + predictions
    )

    confusion_matrix = torch.bincount(
        flat_confusion_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    return {
        "accuracy": correct.float().mean(),
        "macro_accuracy": macro_accuracy,
        "mean_loss": image_losses.mean(),
        "class_accuracy": class_accuracy,
        "class_loss": class_loss,
        "class_counts": class_counts,
        "image_losses": image_losses,
        "predictions": predictions,
        "true_probability": true_class_probability,
        "predicted_probability": predicted_probability,
        "confusion_matrix": confusion_matrix,
    }

def make_confusion_figure(confusion_matrix, class_names):
    """Create a row-normalized confusion_matrix figure."""
    confusion = confusion_matrix.float()

    confusion = confusion / confusion.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1)

    n_classes = len(class_names)
    figure_size = min(18, max(6, n_classes * 0.65))

    fig, ax = plt.subplots(
        figsize=(figure_size, figure_size)
    )

    image = ax.imshow(
        confusion.numpy(),
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    if n_classes <= 30:
        tick_indices = list(range(n_classes))
    else:
        tick_step = math.ceil(n_classes / 20)
        tick_indices = list(range(0, n_classes, tick_step))

    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)

    ax.set_xticklabels(
        [class_names[i] for i in tick_indices],
        rotation=90,
        fontsize=7,
    )

    ax.set_yticklabels(
        [class_names[i] for i in tick_indices],
        fontsize=7,
    )

    ax.set_xlabel("Predicted conformation")
    ax.set_ylabel("True conformation")
    ax.set_title("Confusion matrix against fixed probe set")

    fig.colorbar(image, ax=ax, label="Fraction of true class")
    fig.tight_layout()

    return fig

def make_hard_examples_figure(
    probe_images,
    probe_parameters,
    probe_labels,
    results,
    class_names,
    max_images=16,
):
    """Show the highest-loss misclassified probe images."""
    predictions = results["predictions"]
    losses = results["image_losses"]
    true_probability = results["true_probability"]
    predicted_probability = results["predicted_probability"]

    wrong_indices = torch.where(
        predictions != probe_labels
    )[0]

    # If everything is classified correctly, show the
    # highest-loss correctly classified images instead.
    if wrong_indices.numel() == 0:
        candidate_indices = torch.arange(len(probe_labels))
    else:
        candidate_indices = wrong_indices

    order = torch.argsort(
        losses[candidate_indices],
        descending=True,
    )

    selected = candidate_indices[order[:max_images]]

    n_columns = 4
    n_rows = max(1, math.ceil(len(selected) / n_columns))

    fig, axes_array = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 3.4 * n_rows),
        squeeze=False,
    )

    axes = axes_array.ravel()

    for ax, image_index in zip(axes, selected.tolist()):
        true_class = int(probe_labels[image_index])
        predicted_class = int(predictions[image_index])

        # Simulator parameter positions:
        # 3 = shift
        # 4 = defocus
        # 7 = log10(SNR)
        shift = float(
            probe_parameters[3][image_index].norm()
        )

        defocus = float(
            probe_parameters[4][image_index]
            .reshape(-1)[0]
        )

        log10_snr = float(
            probe_parameters[7][image_index]
            .reshape(-1)[0]
        )

        snr = 10.0 ** log10_snr

        image = (
            probe_images[image_index]
            .detach()
            .cpu()
            .numpy()
        )

        ax.imshow(
            image,
            cmap="gray",
        )

        ax.set_title(
            f"True: {class_names[true_class]}\n"
            f"Pred: {class_names[predicted_class]}\n"
            f"loss={losses[image_index].item():.2f}, "
            f"p(true)={true_probability[image_index].item():.2f}, "
            f"p(pred)={predicted_probability[image_index].item():.2f}\n"
            f"SNR={snr:.3f}, "
            f"defocus={defocus:.2f}, "
            f"shift={shift:.1f} Å",
            fontsize=8,
        )

        ax.axis("off")

    # Hide unused subplot positions.
    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.tight_layout()

    return fig

def summarize_probe_by_snr(
    probe_results,
    probe_labels,
    probe_snr,
    num_structure_classes,
    snr_bins,
):
    """Calculate probe metrics separately for each SNR interval."""

    predictions = probe_results["predictions"]
    image_losses = probe_results["image_losses"]
    correct = predictions.eq(probe_labels)

    summaries = {}

    for bin_index, (lower, upper) in enumerate(snr_bins):
        # Include the upper endpoint only in the final bin.
        if bin_index == len(snr_bins) - 1:
            snr_mask = (
                (probe_snr >= lower)
                & (probe_snr <= upper)
            )
        else:
            snr_mask = (
                (probe_snr >= lower)
                & (probe_snr < upper)
            )

        if not snr_mask.any():
            continue

        # Exclude garbage from structural metrics.
        structure_mask = (
            snr_mask
            & (probe_labels < num_structure_classes)
        )

        structure_labels = probe_labels[structure_mask]
        structure_correct = correct[structure_mask]

        class_counts = torch.bincount(
            structure_labels,
            minlength=num_structure_classes,
        ).float()

        class_correct = torch.bincount(
            structure_labels,
            weights=structure_correct.float(),
            minlength=num_structure_classes,
        )

        valid_classes = class_counts > 0

        class_accuracy = torch.full(
            (num_structure_classes,),
            float("nan"),
        )

        class_accuracy[valid_classes] = (
            class_correct[valid_classes]
            / class_counts[valid_classes]
        )

        tag = f"{lower:.2f}-{upper:.2f}"

        summaries[tag] = {
            # Includes structural and garbage images.
            "all_accuracy": correct[snr_mask].float().mean(),

            # Structural images pooled together.
            "structure_accuracy": (
                structure_correct.float().mean()
            ),

            # Each structural conformation has equal weight.
            "structure_macro_accuracy": (
                class_accuracy[valid_classes].mean()
            ),

            # Accuracy of the most difficult structural class.
            "worst_class_accuracy": (
                class_accuracy[valid_classes].min()
            ),

            "mean_loss": image_losses[snr_mask].mean(),
            "count": int(snr_mask.sum()),
            "structure_count": int(structure_mask.sum()),
            "class_accuracy": class_accuracy,
        }

    return summaries

def log_probe_diagnostics(
    estimator,
    probe_images,
    probe_parameters,
    probe_labels,
    probe_snr,
    simulator,
    class_names,
    num_classes,
    device,
    batch_size,
    writer,
    step,
):
    probe_results = evaluate_probe_set(
        estimator=estimator,
        probe_images=probe_images,
        probe_labels=probe_labels,
        device=device,
        num_classes=num_classes,
        batch_size=batch_size,
    )

    snr_bins = [
        (0.10, 0.20),
        (0.20, 0.30),
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 1.00),
    ]

    snr_results = summarize_probe_by_snr(
        probe_results=probe_results,
        probe_labels=probe_labels,
        probe_snr=probe_snr,
        num_structure_classes=simulator.num_models,
        snr_bins=snr_bins,
    )

    for snr_tag, metrics in snr_results.items():
        writer.add_scalar(
            f"ProbeBySNR/AllAccuracy_{snr_tag}",
            metrics["all_accuracy"].item(),
            step,
        )

        writer.add_scalar(
            f"ProbeBySNR/StructureAccuracy_{snr_tag}",
            metrics["structure_accuracy"].item(),
            step,
        )

        writer.add_scalar(
            f"ProbeBySNR/StructureMacroAccuracy_{snr_tag}",
            metrics["structure_macro_accuracy"].item(),
            step,
        )

        writer.add_scalar(
            f"ProbeBySNR/WorstClassAccuracy_{snr_tag}",
            metrics["worst_class_accuracy"].item(),
            step,
        )

        writer.add_scalar(
            f"ProbeBySNR/MeanLoss_{snr_tag}",
            metrics["mean_loss"].item(),
            step,
        )

        for class_index in range(simulator.num_models):
            class_accuracy = metrics["class_accuracy"][class_index]

            if torch.isfinite(class_accuracy):
                safe_name = class_names[class_index].replace("/", "_")

                writer.add_scalar(
                    f"ProbeBySNRClass/{snr_tag}_{safe_name}",
                    class_accuracy.item(),
                    step,
                )

    writer.add_scalar(
        "Probe/overall_accuracy",
        probe_results["accuracy"].item(),
        step,
    )

    writer.add_scalar(
        "Probe/macro_accuracy",
        probe_results["macro_accuracy"].item(),
        step,
    )

    writer.add_scalar(
        "Probe/mean_loss",
        probe_results["mean_loss"].item(),
        step,
    )

    for class_index, class_name in enumerate(class_names):
        safe_name = class_name.replace("/", "_")

        writer.add_scalar(
            f"ProbeAccuracyByClass/{safe_name}",
            probe_results["class_accuracy"][class_index].item(),
            step,
        )

        writer.add_scalar(
            f"ProbeLossByClass/{safe_name}",
            probe_results["class_loss"][class_index].item(),
            step,
        )

    writer.flush()

    return probe_results

def train_classifier(cfg: DictConfig) -> None:
    """
    Main training function.

    Args:
        cfg: Hydra DictConfig with keys cfg.simulation and cfg.train.
    """
    setup_logging()
    torch.backends.cudnn.benchmark = True

    device = cfg.train.device
    epochs = cfg.train.epochs
    n_workers = cfg.train.n_workers
    saving_frequency = cfg.train.saving_frequency
    simulation_batch_size = cfg.train.simulation_batch_size
    batches_per_epoch = cfg.train.batches_per_epoch
    prefetch_factor = cfg.train.prefetch_factor
    train_from_checkpoint = cfg.train.train_from_checkpoint
    checkpoint_file = cfg.train.get("checkpoint_file", None)

    train_cfg = cfg.train
    image_cfg = cfg.simulation

    batch_size = train_cfg.batch_size
    print(f"simulation_batch_size = {simulation_batch_size}")
    print(f"batch_size = {batch_size}")
    assert simulation_batch_size >= batch_size
    assert simulation_batch_size % batch_size == 0

    simulator = CryoEmSimulator(image_cfg, device=device)

    num_classes = simulator.num_models + (1 if simulator.garbage_class else 0)
    n_reps = simulator.num_representatives if simulator.num_representatives is not None else 1
    suffix = " + garbage class" if simulator.garbage_class else ""
    logging.info(
        f"Training on {simulator.num_models} classes with {n_reps} representatives"
        f"{suffix} (num_classes={num_classes})"
    )

    prior_loader = PriorLoader(
        simulator._priors,
        batch_size=simulation_batch_size,
        num_workers=n_workers,
        prefetch_factor=prefetch_factor,
    )

    if train_from_checkpoint and not checkpoint_file:
        raise ValueError(
            "train.train_from_checkpoint=true but train.checkpoint_file is unset. "
            "Provide a path with train.checkpoint_file=<path>."
        )

    use_amp = bool(getattr(train_cfg, "use_amp", False))
    compile_model = bool(getattr(train_cfg, "compile_model", False))
    is_cuda = str(device).startswith("cuda")
    if use_amp and not is_cuda:
        logging.warning(
            "train.use_amp=true ignored: AMP requires a CUDA device "
            f"(train.device={device})."
        )
        use_amp = False

    estimator = build_classifier(train_cfg, num_classes).to(device=device)
    loss_fn = ClassifierLoss(estimator)

    optimizer = optim.AdamW(
        estimator.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    lr_scheduler = None
    if train_cfg.one_cycle_scheduler:
        logging.info("Using OneCycleLR scheduler")
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg.learning_rate,
            total_steps=epochs * batches_per_epoch * (simulation_batch_size // batch_size),
            pct_start=cfg.train.get("pct_start", 0.3)
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if train_from_checkpoint:
        start_epoch = _load_checkpoint(
            checkpoint_file, estimator, optimizer, lr_scheduler, scaler=scaler
        )
        if start_epoch >= epochs:
            raise ValueError(
                f"checkpoint_file resumes at epoch {start_epoch} but train.epochs={epochs}; "
                "increase train.epochs to continue training."
            )

    if compile_model:
        logging.info("Compiling model with torch.compile")
        estimator = torch.compile(estimator)
        loss_fn = ClassifierLoss(estimator)

    step = GDStep(
        optimizer,
        clip=train_cfg.clip_gradient,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.train.output.tensorboard_dir)

    # ---------------------------------------------------------
    # Fixed diagnostic/probe set
    # ---------------------------------------------------------

    PROBE_EVERY_N_EPOCHS = 5

    probe_size = max(8192, num_classes * 32)

    # Save RNG state so generating the probe set does not
    # change the subsequent training random sequence.
    cpu_rng_state = torch.get_rng_state()

    cuda_rng_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    torch.manual_seed(12345)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    with torch.no_grad():
        probe_images, probe_parameters = (
            simulator.sample_and_simulate(
                num_sim=probe_size,
                return_parameters=True,
                batch_size=simulation_batch_size,
            )
        )
    probe_images = probe_images.detach().cpu()

    probe_parameters = [
        parameter.detach().cpu()
        for parameter in probe_parameters
    ]

    probe_labels = labels_from_parameters(
        probe_parameters,
        simulator,
    ).cpu()

    # Simulator parameter 7 is log10(SNR)
    probe_snr = torch.pow(
        10.0,
        probe_parameters[7].reshape(-1).float()
    )

    # Restore the training RNG state.
    torch.set_rng_state(cpu_rng_state)

    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    class_names = [
        f"conformation_{i:03d}"
        for i in range(simulator.num_models)
    ]

    if simulator.garbage_class:
        class_names.append("garbage")

    probe_class_counts = torch.bincount(
        probe_labels,
        minlength=num_classes,
    )

    class_map_text = "\n".join(
        f"{i}: {name} ({int(probe_class_counts[i])} probe images)"
        for i, name in enumerate(class_names)
    )

    writer.add_text(
        "Probe/class_index_map",
        class_map_text,
        global_step=0,
    )

    logging.info(
        f"Created fixed probe set with {len(probe_images)} images"
    )

    os.makedirs(cfg.train.output.checkpoint_dir, exist_ok=True)
    estimator_file = cfg.train.output.estimator_file
    os.makedirs(os.path.dirname(estimator_file) or ".", exist_ok=True)

    logging.info("Starting training loop")

    log_probe_diagnostics(
        estimator=estimator,
        probe_images=probe_images,
        probe_parameters=probe_parameters,
        probe_labels=probe_labels,
        probe_snr=probe_snr,
        simulator=simulator,
        class_names=class_names,
        num_classes=num_classes,
        device=device,
        batch_size=batch_size,
        writer=writer,
        step=start_epoch,
    )
    start_time = time.time()
    estimator.train()
    final_loss = float("nan")
    final_acc = float("nan")
    global_step = start_epoch * batches_per_epoch * (simulation_batch_size // batch_size)

    with tqdm(range(start_epoch, epochs), unit="epoch", initial=start_epoch, total=epochs) as tq:
        for epoch in tq:
            epoch_losses = []
            epoch_accs = []
            epoch_samples = 0
            epoch_start = time.time()

            for parameters in islice(prior_loader, batches_per_epoch):
                images = simulator.simulate(*parameters)

                # First element is always fg model indices
                fg_indices = parameters[0]
                batch_indices = fg_indices[:, 0] if fg_indices.ndim == 2 else fg_indices
                batch_indices = batch_indices.clone()

                if simulator.garbage_class:
                    garbage_mask = parameters[13]
                    batch_indices[garbage_mask] = simulator.num_models

                for _idx, _img in zip(
                    batch_indices.split(batch_size),
                    images.split(batch_size),
                ):
                    _idx_dev = _idx.to(device, non_blocking=True)
                    _img_dev = _img.to(device, non_blocking=True)

                    with torch.autocast(
                        device_type="cuda" if is_cuda else "cpu",
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        loss, logits = loss_fn(_idx_dev, _img_dev)
                    batch_loss, grad_norm = step(loss)

                    with torch.no_grad():
                        acc = (_idx_dev == logits.argmax(dim=1)).float().mean()

                    writer.add_scalar("Loss/batch", batch_loss.item(), global_step)
                    writer.add_scalar(
                        "LR/step", optimizer.param_groups[0]["lr"], global_step
                    )
                    if grad_norm is not None and torch.isfinite(grad_norm):
                        writer.add_scalar("Gradients/norm", grad_norm.item(), global_step)
                    global_step += 1

                    epoch_losses.append(batch_loss)
                    epoch_accs.append(acc)
                    epoch_samples += _idx.shape[0]

            mean_loss = torch.stack(epoch_losses).mean().item()
            mean_acc = torch.stack(epoch_accs).mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            throughput = epoch_samples / (time.time() - epoch_start)
            writer.add_scalar("Loss/epoch", mean_loss, epoch)
            writer.add_scalar("Accuracy/epoch", mean_acc, epoch)
            writer.add_scalar("LR/epoch", current_lr, epoch)
            writer.add_scalar("Throughput/epoch", throughput, epoch)

            epoch_step = epoch + 1

            if epoch_step % PROBE_EVERY_N_EPOCHS == 0:
                probe_results = log_probe_diagnostics(
                    estimator=estimator,
                    probe_images=probe_images,
                    probe_parameters=probe_parameters,
                    probe_labels=probe_labels,
                    probe_snr=probe_snr,
                    simulator=simulator,
                    class_names=class_names,
                    num_classes=num_classes,
                    device=device,
                    batch_size=batch_size,
                    writer=writer,
                    step=epoch_step,
                )

                # Text table showing the hardest conformations.
                accuracy_for_sorting = torch.nan_to_num(
                    probe_results["class_accuracy"],
                    nan=1.0,
                )

                hardest_classes = torch.argsort(
                    accuracy_for_sorting
                )

                summary_lines = [
                    "| Rank | Conformation | Accuracy | Mean loss | Images |",
                    "|---:|---|---:|---:|---:|",
                ]

                for rank, class_index in enumerate(
                    hardest_classes[: min(10, num_classes)].tolist(),
                    start=1,
                ):
                    summary_lines.append(
                        f"| {rank} | {class_names[class_index]} | "
                        f"{probe_results['class_accuracy'][class_index]:.3f} | "
                        f"{probe_results['class_loss'][class_index]:.3f} | "
                        f"{int(probe_results['class_counts'][class_index])} |"
                    )

                writer.add_text(
                    "Probe/hardest_conformations",
                    "\n".join(summary_lines),
                    epoch_step,
                )

                confusion_figure = make_confusion_figure(
                    probe_results["confusion_matrix"],
                    class_names,
                )

                writer.add_figure(
                    "Probe/confusion_matrix",
                    confusion_figure,
                    epoch_step,
                )

                plt.close(confusion_figure)

                hard_examples_figure = make_hard_examples_figure(
                    probe_images=probe_images,
                    probe_parameters=probe_parameters,
                    probe_labels=probe_labels,
                    results=probe_results,
                    class_names=class_names,
                    max_images=16,
                )

                writer.add_figure(
                    "Probe/hardest_images",
                    hard_examples_figure,
                    epoch_step,
                )

                plt.close(hard_examples_figure)

                writer.flush()
        
            tq.set_postfix(loss=mean_loss, acc=f"{mean_acc:.3f}", lr=current_lr)
            final_loss, final_acc = mean_loss, mean_acc

            # Save after the epoch completes; (epoch+1) so we never save an
            # untrained model at epoch 0 and we always save the final epoch.
            if (
                (epoch + 1) % saving_frequency == 0
                or (epoch + 1) == epochs
            ):
                ckpt_path = os.path.join(
                    cfg.train.output.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
                )
                _save_checkpoint(
                    ckpt_path, estimator, optimizer, lr_scheduler, epoch + 1, scaler=scaler
                )

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Final estimator: weights only — used by classifier_utils.load_classifier
    # at inference time. Periodic checkpoints (above) carry the full state.
    torch.save(_underlying_module(estimator).state_dict(), estimator_file)

    # Bind hparams to this run with the final metrics so they share a TB run dir.
    hparams = OmegaConf.to_container(cfg.train, resolve=True)
    flat_hparams = {
        k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparams.items()
    }
    writer.add_hparams(
        flat_hparams,
        metric_dict={"final/loss": final_loss, "final/acc": final_acc},
    )
    writer.close()
