from __future__ import annotations

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from src.base.shared_utils import _print


class ProgressCallback(TrainerCallback):
    """Lightweight training-progress logger.

    ``SlimTrainer`` wires a reference to itself into ``self._trainer`` after
    construction, so this callback can read the per-step loss components
    (``loss_task``/``loss_kd``/``loss_logits``) that ``SlimTrainer.compute_loss``
    stashes on the trainer. All fields are optional and accessed defensively so
    the callback stays a no-op when they are absent.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._trainer = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not self.verbose or not state.is_world_process_zero:
            return

        logs = logs or {}
        loss = logs.get("loss")
        parts = [f"step {state.global_step}"]
        if state.max_steps:
            parts[0] += f"/{state.max_steps}"
        if loss is not None:
            parts.append(f"loss={loss:.4f}")

        trainer = self._trainer
        if trainer is not None:
            for name in ("loss_task", "loss_kd", "loss_logits"):
                val = getattr(trainer, name, None)
                if val:
                    parts.append(f"{name}={float(val):.4f}")

        _print("[Progress] " + " | ".join(parts))
