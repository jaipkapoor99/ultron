import json
import os
from typing import Any

import torch

from telemetry import UltronTelemetry


class UltronTrainer:
    def __init__(
        self,
        model: Any,
        optimizer_muon: Any,
        optimizer_adamw: Any,
        train_loader: Any,
        dev_loader: Any,
        config: Any,
        accelerator: Any,
    ) -> None:
        self.model = model
        self.optimizer_muon = optimizer_muon
        self.optimizer_adamw = optimizer_adamw
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.accelerator: Any = accelerator

        self.accelerate_dir = "accelerate_checkpoint"
        self.step = 0
        self.dev_batch_cursor = 0
        self.decay_start_step = int(0.8 * config.max_steps)
        task_type = "sft" if "sft" in self.accelerate_dir else "pretrain"
        self.telemetry: Any = UltronTelemetry(
            config,
            accelerator,
            checkpoint_dir=self.accelerate_dir,
            task_type=task_type,
        )
        self.tokens_per_step = self.telemetry.global_tokens_per_step
        self.total_training_tokens = self.tokens_per_step * config.max_steps
        if len(self.train_loader) % accelerator.gradient_accumulation_steps:
            raise ValueError(
                "Training dataloader batches per epoch must be divisible by "
                "gradient_accumulation_steps for exact resume"
            )

    def print_rich(self, msg: str) -> None:
        if hasattr(self, "telemetry") and self.telemetry is not None:
            self.telemetry.print_message(msg)
        else:
            self.accelerator.print(msg)

    def print_table_row(
        self, step: int, train_loss: float, dev_loss: float, lr: float
    ) -> None:
        """Print one compact evaluation summary through the telemetry UI."""
        self.print_rich(
            f"Step {step:,} | train loss {train_loss:.4f} | "
            f"sampled dev loss {dev_loss:.4f} | lr {lr:.3e}"
        )

    def update_learning_rate(self) -> float:
        # WSD (Warmup-Stable-Linear-Decay) Learning Rate Schedule
        if self.step < self.config.warmup_steps:
            lr = self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        elif self.step < self.decay_start_step:
            lr = self.config.learning_rate
        else:
            decay_ratio = (self.step - self.decay_start_step) / max(
                1, self.config.max_steps - self.decay_start_step
            )
            lr = self.config.min_lr + (1.0 - decay_ratio) * (
                self.config.learning_rate - self.config.min_lr
            )

        for param_group in self.optimizer_adamw.param_groups:
            param_group["lr"] = lr

        if self.optimizer_muon is not None:
            for param_group in self.optimizer_muon.param_groups:
                param_group["lr"] = 0.04 * (lr / self.config.learning_rate)
        return lr

    def load_checkpoint(self) -> None:
        if os.path.isdir(self.accelerate_dir):
            self.accelerator.print(
                f"Resuming training state from '{self.accelerate_dir}'..."
            )
            self.accelerator.load_state(self.accelerate_dir)

            state_file = os.path.join(self.accelerate_dir, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file) as f:
                    state_info = json.load(f)
                    checkpoint_seed = state_info.get("data_seed")
                    if (
                        checkpoint_seed is not None
                        and checkpoint_seed != self.config.data_seed
                    ):
                        raise RuntimeError(
                            "Checkpoint data_seed does not match the current "
                            "configuration"
                        )
                    self.step = state_info.get("step", 0)
                    self.dev_batch_cursor = state_info.get("dev_batch_cursor", 0) % max(
                        1, len(self.dev_loader)
                    )
                    self.accelerator.print(f"✓ Restored training step: {self.step:,}")
            self.accelerator.print("✓ State restored successfully!")
        else:
            self.accelerator.print(
                f"⚠ No checkpoint found at '{self.accelerate_dir}', starting from scratch."
            )

    def _sample_dev_batches(self) -> Any:
        """Yield the next deterministic window of validation batches."""
        total_batches = len(self.dev_loader)
        if total_batches == 0:
            raise RuntimeError("Validation dataloader contains no batches")

        target_batches = min(self.config.eval_batches, total_batches)
        remaining = target_batches
        while remaining > 0:
            available = total_batches - self.dev_batch_cursor
            segment_size = min(remaining, available)
            segment_loader = self.dev_loader
            if self.dev_batch_cursor:
                segment_loader = self.accelerator.skip_first_batches(
                    self.dev_loader,
                    self.dev_batch_cursor,
                )

            consumed = 0
            for batch in segment_loader:
                yield batch
                consumed += 1
                self.dev_batch_cursor = (self.dev_batch_cursor + 1) % total_batches
                if consumed >= segment_size:
                    break
            if consumed != segment_size:
                raise RuntimeError(
                    "Validation dataloader ended before the sampled window was complete"
                )
            remaining -= consumed

    def evaluate(self, train_loss: float, lr: float) -> None:
        self.model.eval()
        total_dev_loss = torch.zeros(
            (), device=self.accelerator.device, dtype=torch.float64
        )
        total_dev_tokens = torch.zeros(
            (), device=self.accelerator.device, dtype=torch.float64
        )
        dev_batches = 0
        with torch.no_grad():
            for xb_dev, yb_dev in self._sample_dev_batches():
                dev_out = self.model(xb_dev, yb_dev)
                dev_loss = (
                    dev_out.loss
                    if (hasattr(dev_out, "loss") and dev_out.loss is not None)
                    else dev_out[1]
                )
                token_count = yb_dev.numel()
                total_dev_loss += dev_loss.detach().double() * token_count
                total_dev_tokens += token_count
                dev_batches += 1

        totals = self.accelerator.reduce(
            torch.stack((total_dev_loss, total_dev_tokens)),
            reduction="sum",
        )
        if totals[1].item() == 0:
            raise RuntimeError("Validation dataloader produced no tokens")
        avg_dev_loss = (totals[0] / totals[1]).item()

        self.telemetry.log_evaluation(
            step=self.step,
            train_loss=train_loss,
            dev_loss=avg_dev_loss,
            lr=lr,
        )
        self.save_checkpoint()
        self.model.train()

    def save_checkpoint(self, final: bool = False) -> None:
        if getattr(self.config, "is_test_mode", False):
            return
        # Save only the Accelerate training state (model weights, optimizer, etc.)
        self.accelerator.save_state(self.accelerate_dir)
        self.accelerator.wait_for_everyone()

        # Persist the current step and wandb run ID & name so we can resume correctly
        state_payload = {
            "step": self.step,
            "max_steps": self.config.max_steps,
            "data_seed": self.config.data_seed,
            "dev_batch_cursor": self.dev_batch_cursor,
            "model_config": self.config.to_metadata(),
        }
        run_id = getattr(self.telemetry, "get_wandb_run_id", lambda: None)()
        run_name = getattr(self.telemetry, "get_wandb_run_name", lambda: None)()
        if run_id:
            state_payload["wandb_run_id"] = run_id
        if run_name:
            state_payload["wandb_run_name"] = run_name

        if self.accelerator.is_main_process:
            state_file = os.path.join(self.accelerate_dir, "training_state.json")
            temporary_state_file = f"{state_file}.tmp"
            with open(temporary_state_file, "w") as f:
                json.dump(state_payload, f, indent=2)
            os.replace(temporary_state_file, state_file)
        self.accelerator.wait_for_everyone()

        if final:
            self.print_rich("✓ Saved Accelerate checkpoint.")

    def train(self) -> None:
        self.model.train()

        label = (
            "Supervised Fine-Tuning" if "sft" in self.accelerate_dir else "Pre-training"
        )
        self.print_rich(
            f"[bold yellow]⚡ {label} for {self.config.max_steps:,} steps ({self.total_training_tokens:,} total tokens)...[/bold yellow]\n"
        )
        # Training loop without tqdm progress bar

        data_epoch, skip_count = self._data_position(self.step)

        if self.step > 0:
            self.print_rich(
                f"[bold yellow]⏩ Fast-forwarding dataset past {skip_count:,} batches...[/bold yellow]"
            )
            active_dataloader = self.accelerator.skip_first_batches(
                self.train_loader,
                skip_count,
            )
        else:
            active_dataloader = self.train_loader
        if hasattr(active_dataloader, "set_epoch"):
            active_dataloader.set_epoch(data_epoch)

        while self.step < self.config.max_steps:
            for xb, yb in active_dataloader:
                if self.step >= self.config.max_steps:
                    break

                lr = self.update_learning_rate()

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        out = self.model(xb, yb)
                        if hasattr(out, "loss") and out.loss is not None:
                            loss = out.loss
                        else:
                            loss = out[1]

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer_adamw.step()
                    if self.optimizer_muon is not None:
                        self.optimizer_muon.step()

                    self.optimizer_adamw.zero_grad()
                    if self.optimizer_muon is not None:
                        self.optimizer_muon.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.step += 1
                        self.telemetry.update_terminal_progress(
                            self.step,
                            loss=loss.item(),
                        )
                        is_eval_step = (
                            self.step % self.config.eval_interval == 0
                            or self.step == self.config.max_steps
                        )
                        if not is_eval_step:
                            # Non-eval steps: log train metrics only
                            self.telemetry.log_training_step(
                                step=self.step,
                                loss=loss.item(),
                                lr=lr,
                            )
                        if is_eval_step:
                            self.evaluate(loss.item(), lr)
            data_epoch += 1
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(data_epoch)
            active_dataloader = self.train_loader

        self.telemetry.close()
        self.print_rich("\n[bold green]🎉 Pre-training Complete![/bold green]")
        self.save_checkpoint(final=True)

    def _data_position(self, step: int) -> tuple[int, int]:
        """Return deterministic shuffle epoch and batch offset for a step."""
        if step < 0:
            raise ValueError("step cannot be negative")
        consumed_batches = step * self.accelerator.gradient_accumulation_steps
        return divmod(consumed_batches, len(self.train_loader))
