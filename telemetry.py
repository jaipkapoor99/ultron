"""Training and tokenization telemetry with rolling-rate estimates."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import warnings
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from tqdm import tqdm

Clock = Callable[[], float]


def wandb_run_name(
    task_type: str = "pretrain",
    now: datetime | None = None,
) -> str:
    """Return a timestamped run name for a fresh training or fine-tuning run."""
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    custom_name = os.environ.get("ULTRON_RUN_NAME")
    if custom_name:
        return f"{timestamp}-{custom_name}"
    suffix = "sft" if task_type == "sft" else "pretrain"
    return f"{timestamp}-ultron-113m-{suffix}"


@dataclass(frozen=True)
class RateSnapshot:
    """A point-in-time rolling-rate estimate."""

    units_per_second: float = 0.0
    elapsed_seconds: float = 0.0


class RollingRateMeter:
    """Estimate recent throughput from monotonic cumulative counters."""

    def __init__(
        self, window_seconds: float = 30.0, clock: Clock = time.monotonic
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        self.window_seconds = window_seconds
        self.clock = clock
        self.samples: deque[tuple[float, float]] = deque()

    def update(self, total_units: float) -> RateSnapshot:
        now = self.clock()
        if self.samples and total_units < self.samples[-1][1]:
            self.samples.clear()
        self.samples.append((now, total_units))

        while len(self.samples) > 2 and now - self.samples[0][0] > self.window_seconds:
            self.samples.popleft()

        first_time, first_units = self.samples[0]
        elapsed = now - first_time
        if elapsed <= 0 or len(self.samples) < 2:
            return RateSnapshot(elapsed_seconds=max(0.0, elapsed))
        rate = max(0.0, (total_units - first_units) / elapsed)
        return RateSnapshot(units_per_second=rate, elapsed_seconds=elapsed)


def format_rate(rate: float, unit: str) -> str:
    """Format a non-negative rate with compact SI units."""
    if not math.isfinite(rate) or rate <= 0:
        return f"— {unit}/s"
    if rate >= 1e9:
        return f"{rate / 1e9:.2f}G {unit}/s"
    if rate >= 1e6:
        return f"{rate / 1e6:.2f}M {unit}/s"
    if rate >= 1e3:
        return f"{rate / 1e3:.1f}k {unit}/s"
    return f"{rate:.1f} {unit}/s"


class UltronTelemetry:
    """W&B metrics, rolling throughput, ETA, and minimal terminal progress."""

    PROJECT_NAME = "ultron-pretraining"
    TRAIN_LOSS_METRIC = "train/loss"
    AVERAGE_TRAIN_LOSS_METRIC = "train/average_loss"
    DEV_LOSS_METRIC = "eval/dev_loss"
    SAMPLED_DEV_LOSS_METRIC = "eval/sampled_dev_loss"
    LOSS_COMPARISON_CHART = "train_vs_dev_loss"
    TOKENS_PER_SECOND_METRIC = "perf/tokens_per_sec"
    STEPS_PER_SECOND_METRIC = "perf/steps_per_sec"
    RATE_WINDOW_SECONDS = 30.0
    RENDER_INTERVAL_SECONDS = 0.5

    def __init__(
        self,
        config: Any,
        accelerator: Any,
        checkpoint_dir: str = "accelerate_checkpoint",
        *,
        clock: Clock = time.monotonic,
        task_type: str = "pretrain",
    ) -> None:
        self.config = config
        self.accelerator = accelerator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.clock = clock
        self.task_type = task_type
        self.rate_meter = RollingRateMeter(self.RATE_WINDOW_SECONDS, clock)
        self.pbar: tqdm | None = None
        self.last_render_time = float("-inf")
        self.last_dev_loss: float | None = None
        self.last_throughput = 0.0
        self.last_steps_per_sec = 0.0
        self.last_eta_seconds = 0
        self._tracker_warning_emitted = False
        self._chart_warning_emitted = False
        self._train_loss_total = 0.0
        self._train_loss_samples = 0
        self._loss_history_steps: list[int] = []
        self._average_train_loss_history: list[float] = []
        self._dev_loss_history: list[float] = []
        self.best_dev_loss = float("inf")
        self.best_average_train_loss = float("inf")

    @classmethod
    def setup_accelerator_trackers(
        cls,
        config: Any,
        args: Any,
        checkpoint_dir: str = "accelerate_checkpoint",
        *,
        task_type: str = "pretrain",
        project_name: str | None = None,
    ) -> Accelerator:
        """Create Accelerator and configure a resumable W&B tracker with persistent run ID and name."""
        is_test = getattr(args, "mode", None) == "test" or getattr(
            config, "is_test_mode", False
        )
        if is_test:
            return Accelerator(
                gradient_accumulation_steps=config.grad_accum_steps,
                log_with=None,
            )

        run_mode = getattr(args, "mode", "continue")
        state_path = Path(checkpoint_dir) / "training_state.json"

        saved_run_id: str | None = None
        saved_run_name: str | None = None
        if run_mode == "continue" and state_path.exists():
            try:
                with state_path.open() as file:
                    state_data = json.load(file)
                    saved_run_id = state_data.get("wandb_run_id")
                    saved_run_name = state_data.get("wandb_run_name")
            except (OSError, json.JSONDecodeError) as error:
                warnings.warn(
                    f"Could not read W&B resume metadata from {state_path}: {error}",
                    stacklevel=2,
                )

        if saved_run_id:
            wandb_options: dict[str, Any] = {
                "id": saved_run_id,
                "resume": "must",
            }
            if saved_run_name:
                wandb_options["name"] = saved_run_name
        else:
            fresh_name = wandb_run_name(task_type)
            wandb_options = {
                "name": fresh_name,
                "resume": "never" if run_mode == "fresh" else "allow",
            }

        accelerator = Accelerator(
            gradient_accumulation_steps=config.grad_accum_steps,
            log_with="wandb",
        )
        target_project = project_name or (
            "ultron-sft" if task_type == "sft" else cls.PROJECT_NAME
        )
        accelerator.init_trackers(
            target_project,
            config=dataclasses.asdict(config),
            init_kwargs={"wandb": wandb_options},
        )
        cls._define_wandb_metrics(accelerator)
        cls._initialize_wandb_summary(config, accelerator)
        return accelerator

    @staticmethod
    def _wandb_run(accelerator: Any) -> Any:
        if not accelerator.is_main_process:
            return None
        return accelerator.get_tracker("wandb", unwrap=True)

    @classmethod
    def _initialize_wandb_summary(cls, config: Any, accelerator: Any) -> None:
        """Populate immutable run totals before the first metric is logged."""
        try:
            run: Any = cls._wandb_run(accelerator=accelerator)
            if run is None:
                return
            tokens_per_step = (
                config.B
                * accelerator.gradient_accumulation_steps
                * config.T
                * accelerator.num_processes
            )
            run.summary.update(
                {
                    "training/status": "running",
                    "training/current_step": 0,
                    "training/total_steps": config.max_steps,
                    "training/tokens_per_step": tokens_per_step,
                    "training/tokens_processed": 0,
                    "training/total_tokens": tokens_per_step * config.max_steps,
                    "training/progress_percent": 0.0,
                }
            )
        except (AttributeError, KeyError, RuntimeError, ValueError) as error:
            warnings.warn(
                message=f"W&B summary could not be initialized: {error}",
                stacklevel=2,
            )

    @staticmethod
    def _define_wandb_metrics(accelerator: Any) -> None:
        if not accelerator.is_main_process:
            return
        try:
            import wandb

            wandb.define_metric(name="train/*")
            wandb.define_metric("eval/*")
            wandb.define_metric("perf/*")
            wandb.define_metric(
                UltronTelemetry.TRAIN_LOSS_METRIC,
                summary="last",
            )
            wandb.define_metric(
                UltronTelemetry.AVERAGE_TRAIN_LOSS_METRIC,
                summary="min",
            )
            wandb.define_metric(
                UltronTelemetry.DEV_LOSS_METRIC,
                summary="min",
            )
            wandb.define_metric(
                UltronTelemetry.SAMPLED_DEV_LOSS_METRIC,
                hidden=True,
                summary="min",
            )
        except (ImportError, RuntimeError) as error:
            warnings.warn(
                f"W&B metric definitions were not applied: {error}",
                stacklevel=2,
            )

    @property
    def global_tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step across all workers."""
        return (
            self.config.B
            * self.accelerator.gradient_accumulation_steps
            * self.config.T
            * self.accelerator.num_processes
        )

    def get_wandb_run_id(self) -> str | None:
        """Return the active W&B run ID, warning once on tracker failures."""
        if not self.accelerator.is_main_process:
            return None
        try:
            run = self._wandb_run(self.accelerator)
            return getattr(run, "id", None)
        except (KeyError, RuntimeError, AttributeError) as error:
            if not self._tracker_warning_emitted:
                warnings.warn(
                    f"Could not resolve the active W&B run ID: {error}",
                    stacklevel=2,
                )
                self._tracker_warning_emitted = True
            return None

    def get_wandb_run_name(self) -> str | None:
        """Return the active W&B run name, warning once on tracker failures."""
        if not self.accelerator.is_main_process:
            return None
        try:
            run = self._wandb_run(self.accelerator)
            return getattr(run, "name", None)
        except KeyError, RuntimeError, AttributeError:
            return None

    def _init_pbar(self, initial_step: int) -> None:
        if self.accelerator.is_main_process and self.pbar is None:
            desc = "🎯 SFT" if self.task_type == "sft" else "⚡ Pre-training"
            self.pbar = tqdm(
                total=self.config.max_steps,
                initial=initial_step,
                desc=desc,
                unit="step",
                dynamic_ncols=True,
                leave=True,
                mininterval=self.RENDER_INTERVAL_SECONDS,
            )

    def set_last_dev_loss(self, dev_loss: float) -> None:
        self.last_dev_loss = dev_loss

    def update_terminal_progress(
        self,
        current_step: int,
        loss: float | None = None,
    ) -> int:
        """Update rolling global throughput and the main-process progress bar."""
        snapshot = self.rate_meter.update(current_step)
        steps_per_second = snapshot.units_per_second
        tokens_per_second = steps_per_second * self.global_tokens_per_step
        remaining_steps = max(0, self.config.max_steps - current_step)
        eta_seconds = (
            int(remaining_steps / steps_per_second) if steps_per_second > 0 else 0
        )

        self.last_steps_per_sec = steps_per_second
        self.last_throughput = tokens_per_second
        self.last_eta_seconds = eta_seconds

        if self.pbar is None:
            self._init_pbar(current_step)
        now = self.clock()
        should_render = (
            current_step >= self.config.max_steps
            or now - self.last_render_time >= self.RENDER_INTERVAL_SECONDS
        )
        if self.accelerator.is_main_process and self.pbar is not None and should_render:
            self.pbar.n = current_step
            postfix = {
                "tok/s": format_rate(tokens_per_second, "tok"),
                "ETA": f"{eta_seconds // 60:d}m" if eta_seconds else "—",
            }
            if loss is not None:
                postfix["train_loss"] = f"{loss:.4f}"
            if self.last_dev_loss is not None:
                postfix["sampled_dev_loss"] = f"{self.last_dev_loss:.4f}"
            self.pbar.set_postfix(postfix, refresh=True)
            self.last_render_time = now
        return eta_seconds

    def print_message(self, text: str) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.pbar is not None:
            self.pbar.write(text)
        else:
            self.accelerator.print(text)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def log_step(self, metrics: Mapping[str, Any], step: int) -> None:
        self.accelerator.log(dict(metrics), step=step)

    def _performance_metrics(self) -> dict[str, float]:
        return {
            self.TOKENS_PER_SECOND_METRIC: self.last_throughput,
            self.STEPS_PER_SECOND_METRIC: self.last_steps_per_sec,
        }

    def _update_wandb_summary(self, values: Mapping[str, Any]) -> None:
        """Update summary-only values without creating dashboard time series."""
        if not self.accelerator.is_main_process:
            return
        try:
            run: Any = self._wandb_run(self.accelerator)
            if run is not None:
                run.summary.update(dict(values))
        except (AttributeError, KeyError, RuntimeError, ValueError) as error:
            if not self._tracker_warning_emitted:
                warnings.warn(
                    f"W&B summary could not be updated: {error}",
                    stacklevel=2,
                )
                self._tracker_warning_emitted = True

    def _live_summary(
        self,
        step: int,
        loss: float,
        lr: float,
    ) -> dict[str, float | int]:
        return {
            "training/current_step": step,
            "training/tokens_processed": step * self.global_tokens_per_step,
            "training/progress_percent": 100.0 * step / self.config.max_steps,
            "latest/train_loss": loss,
            "latest/learning_rate": lr,
            "latest/tokens_per_sec": self.last_throughput,
            "latest/steps_per_sec": self.last_steps_per_sec,
        }

    def _record_train_loss(self, loss: float) -> None:
        if math.isfinite(loss):
            self._train_loss_total += loss
            self._train_loss_samples += 1

    def _finish_train_loss_interval(self, fallback: float) -> float:
        if self._train_loss_samples == 0:
            return fallback
        average = self._train_loss_total / self._train_loss_samples
        self._train_loss_total = 0.0
        self._train_loss_samples = 0
        return average

    def _build_loss_comparison_chart(self) -> Any:
        """Build the compact train-average versus sampled-dev W&B chart."""
        if not self.accelerator.is_main_process:
            return None
        try:
            import wandb

            return wandb.plot.line_series(
                xs=self._loss_history_steps,
                ys=[
                    self._average_train_loss_history,
                    self._dev_loss_history,
                ],
                keys=["Average train loss", "Sampled dev loss"],
                title="Average Train Loss vs Sampled Dev Loss",
                xname="Optimizer step",
            )
        except (ImportError, IndexError, RuntimeError, TypeError, ValueError) as error:
            if not self._chart_warning_emitted:
                warnings.warn(
                    f"W&B loss comparison chart could not be built: {error}",
                    stacklevel=2,
                )
                self._chart_warning_emitted = True
            return None

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
    ) -> None:
        self._record_train_loss(loss)
        metrics = {
            self.TRAIN_LOSS_METRIC: loss,
            "train/lr": lr,
            **self._performance_metrics(),
        }
        if self.last_dev_loss is not None:
            # Hold the latest sampled estimate between validation passes so the
            # dashboard renders a continuous, explicitly sampled dev-loss line.
            metrics[self.DEV_LOSS_METRIC] = self.last_dev_loss
        self.log_step(metrics, step)
        self._update_wandb_summary(self._live_summary(step, loss, lr))

    def log_evaluation(
        self,
        step: int,
        train_loss: float,
        dev_loss: float,
        lr: float,
    ) -> None:
        """Log a deliberately sampled validation estimate."""
        self._record_train_loss(train_loss)
        average_train_loss = self._finish_train_loss_interval(train_loss)
        self.set_last_dev_loss(dev_loss)
        self._loss_history_steps.append(step)
        self._average_train_loss_history.append(average_train_loss)
        self._dev_loss_history.append(dev_loss)
        self.best_dev_loss = min(self.best_dev_loss, dev_loss)
        self.best_average_train_loss = min(
            self.best_average_train_loss,
            average_train_loss,
        )

        metrics: dict[str, Any] = {
            self.DEV_LOSS_METRIC: dev_loss,
            self.SAMPLED_DEV_LOSS_METRIC: dev_loss,
            self.TRAIN_LOSS_METRIC: train_loss,
            self.AVERAGE_TRAIN_LOSS_METRIC: average_train_loss,
            "train/lr": lr,
            **self._performance_metrics(),
        }
        comparison_chart = self._build_loss_comparison_chart()
        if comparison_chart is not None:
            metrics[self.LOSS_COMPARISON_CHART] = comparison_chart
        self.log_step(metrics, step)
        self._update_wandb_summary(
            {
                **self._live_summary(step, train_loss, lr),
                "latest/average_train_loss": average_train_loss,
                "latest/dev_loss": dev_loss,
                "best/average_train_loss": self.best_average_train_loss,
                "best/dev_loss": self.best_dev_loss,
                "validation/evaluations": len(self._dev_loss_history),
            }
        )


class TokenizationTelemetry:
    """Rolling throughput and throttled progress for dataset tokenization."""

    RATE_WINDOW_SECONDS = 30.0
    RENDER_INTERVAL_SECONDS = 0.5

    def __init__(
        self,
        target_tokens: int | None = None,
        start_tokens: int = 0,
        *,
        clock: Clock = time.monotonic,
        enabled: bool = True,
    ) -> None:
        if target_tokens is not None and target_tokens <= 0:
            raise ValueError("target_tokens must be greater than zero")
        if start_tokens < 0 or (
            target_tokens is not None and start_tokens > target_tokens
        ):
            raise ValueError("start_tokens must be within the target range")
        self.target_tokens = target_tokens
        self.current_total = start_tokens
        self.clock = clock
        self.rate_meter = RollingRateMeter(self.RATE_WINDOW_SECONDS, clock)
        self.rate_meter.update(start_tokens)
        self.last_render_time = float("-inf")
        self.last_tokens_per_second = 0.0
        self.last_eta_seconds = 0
        self.pbar: tqdm | None = None
        if enabled:
            self.pbar = tqdm(
                total=target_tokens,
                initial=start_tokens,
                desc="📦 Tokenizing",
                unit="tok",
                unit_scale=True,
                dynamic_ncols=True,
                leave=True,
                mininterval=self.RENDER_INTERVAL_SECONDS,
            )

    def update(
        self,
        added_tokens: int,
        current_total: int,
        shard_info: str | None = None,
    ) -> int:
        if added_tokens < 0:
            raise ValueError("added_tokens cannot be negative")
        if current_total < 0:
            raise ValueError("current_total cannot be negative")
        observed_delta = current_total - self.current_total
        if observed_delta != added_tokens:
            raise ValueError(
                "added_tokens must equal the change in current_total "
                f"({added_tokens} != {observed_delta})"
            )
        self.current_total = current_total

        snapshot = self.rate_meter.update(current_total)
        tokens_per_second = snapshot.units_per_second
        if self.target_tokens is not None:
            remaining_tokens = max(0, self.target_tokens - current_total)
            eta_seconds = (
                int(remaining_tokens / tokens_per_second)
                if tokens_per_second > 0
                else 0
            )
        else:
            eta_seconds = 0
        self.last_tokens_per_second = tokens_per_second
        self.last_eta_seconds = eta_seconds

        now = self.clock()
        should_render = (
            self.target_tokens is not None and current_total >= self.target_tokens
        ) or now - self.last_render_time >= self.RENDER_INTERVAL_SECONDS
        if self.pbar is not None and should_render:
            self.pbar.n = (
                min(current_total, self.target_tokens)
                if self.target_tokens is not None
                else current_total
            )
            postfix = {
                "tok/s": format_rate(tokens_per_second, "tok"),
            }
            if self.target_tokens is not None and eta_seconds:
                postfix["ETA"] = f"{eta_seconds // 60:d}m"
            if shard_info:
                postfix["shard"] = shard_info
            self.pbar.set_postfix(postfix, refresh=True)
            self.last_render_time = now
        return eta_seconds

    def print_message(self, text: str) -> None:
        if self.pbar is not None:
            self.pbar.write(text)
        else:
            print(text)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class ValidationTelemetry:
    """W&B-backed progress and timing for complete validation passes."""

    PROJECT_NAME = UltronTelemetry.PROJECT_NAME
    LOSS_METRIC = "validation/loss"
    TOKENS_PER_SECOND_METRIC = "validation/tokens_per_sec"
    PROGRESS_METRIC = "validation/progress_percent"
    RATE_WINDOW_SECONDS = 30.0
    RENDER_INTERVAL_SECONDS = 0.5

    @classmethod
    def setup_accelerator(
        cls,
        config: Any,
        *,
        project_name: str | None = None,
        run_name: str | None = None,
    ) -> Accelerator:
        """Create a fresh W&B run dedicated to one full validation pass."""
        accelerator = Accelerator(log_with="wandb")
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        accelerator.init_trackers(
            project_name or cls.PROJECT_NAME,
            config=dataclasses.asdict(config),
            init_kwargs={
                "wandb": {
                    "name": run_name or f"{timestamp}-full-validation",
                    "job_type": "full-validation",
                    "resume": "never",
                    "tags": ["full-validation"],
                }
            },
        )
        cls._define_wandb_metrics(accelerator)
        return accelerator

    @classmethod
    def _define_wandb_metrics(cls, accelerator: Any) -> None:
        if not accelerator.is_main_process:
            return
        try:
            import wandb

            wandb.define_metric("validation/*")
            wandb.define_metric(cls.LOSS_METRIC, summary="min")
            wandb.define_metric(cls.TOKENS_PER_SECOND_METRIC, summary="max")
        except (ImportError, RuntimeError) as error:
            warnings.warn(
                f"W&B validation metric definitions were not applied: {error}",
                stacklevel=2,
            )

    def __init__(
        self,
        total_sequences: int,
        sequence_length: int,
        accelerator: Any,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        if total_sequences <= 0:
            raise ValueError("total_sequences must be greater than zero")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        self.total_sequences = total_sequences
        self.sequence_length = sequence_length
        self.accelerator = accelerator
        self.clock = clock
        self.start_time = clock()
        self.end_time: float | None = None
        self.processed_sequences = 0
        self.rate_meter = RollingRateMeter(self.RATE_WINDOW_SECONDS, clock)
        self.rate_meter.update(0)
        self.last_render_time = float("-inf")
        self.last_tokens_per_second = 0.0
        self.last_eta_seconds = 0
        self._tracker_warning_emitted = False
        self.pbar: tqdm | None = None
        if accelerator.is_main_process:
            self.pbar = tqdm(
                total=total_sequences,
                desc="🔎 Full validation",
                unit="seq",
                dynamic_ncols=True,
                leave=True,
                mininterval=self.RENDER_INTERVAL_SECONDS,
            )
            self._update_wandb_summary(
                {
                    "validation/status": "running",
                    "validation/total_sequences": total_sequences,
                    "validation/total_tokens": (total_sequences * sequence_length),
                    "validation/sequences_processed": 0,
                    "validation/tokens_processed": 0,
                    "validation/progress_percent": 0.0,
                }
            )

    def _update_wandb_summary(self, values: Mapping[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        try:
            run: Any = self.accelerator.get_tracker("wandb", unwrap=True)
            if run is not None:
                run.summary.update(dict(values))
        except (AttributeError, KeyError, RuntimeError, ValueError) as error:
            if not self._tracker_warning_emitted:
                warnings.warn(
                    f"W&B validation summary could not be updated: {error}",
                    stacklevel=2,
                )
                self._tracker_warning_emitted = True

    @property
    def elapsed_seconds(self) -> float:
        end_time = self.end_time if self.end_time is not None else self.clock()
        return max(0.0, end_time - self.start_time)

    @property
    def average_tokens_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.processed_sequences * self.sequence_length / elapsed

    def update(self, processed_sequences: int, mean_loss: float) -> int:
        if processed_sequences < self.processed_sequences:
            raise ValueError("processed_sequences cannot move backwards")
        self.processed_sequences = processed_sequences
        processed_tokens = processed_sequences * self.sequence_length
        snapshot = self.rate_meter.update(processed_tokens)
        tokens_per_second = snapshot.units_per_second
        remaining_tokens = max(
            0,
            (self.total_sequences - processed_sequences) * self.sequence_length,
        )
        eta_seconds = (
            int(remaining_tokens / tokens_per_second) if tokens_per_second > 0 else 0
        )
        self.last_tokens_per_second = tokens_per_second
        self.last_eta_seconds = eta_seconds

        now = self.clock()
        should_render = (
            processed_sequences >= self.total_sequences
            or now - self.last_render_time >= self.RENDER_INTERVAL_SECONDS
        )
        if self.accelerator.is_main_process and should_render:
            progress_percent = 100.0 * processed_sequences / self.total_sequences
            self.accelerator.log(
                {
                    self.LOSS_METRIC: mean_loss,
                    self.TOKENS_PER_SECOND_METRIC: tokens_per_second,
                    self.PROGRESS_METRIC: progress_percent,
                },
                step=processed_sequences,
            )
            self._update_wandb_summary(
                {
                    "validation/sequences_processed": processed_sequences,
                    "validation/tokens_processed": processed_tokens,
                    "validation/progress_percent": progress_percent,
                    "latest/validation_loss": mean_loss,
                    "latest/validation_tokens_per_sec": tokens_per_second,
                }
            )
            if self.pbar is not None:
                self.pbar.n = min(processed_sequences, self.total_sequences)
                self.pbar.set_postfix(
                    {
                        "loss": f"{mean_loss:.4f}",
                        "tok/s": format_rate(tokens_per_second, "tok"),
                        "ETA": f"{eta_seconds // 60:d}m" if eta_seconds else "—",
                    },
                    refresh=True,
                )
            self.last_render_time = now
        return eta_seconds

    def finish(self, *, loss: float, perplexity: float) -> None:
        """Finalize W&B result fields after a successful complete pass."""
        self._update_wandb_summary(
            {
                "validation/status": "complete",
                "validation/loss": loss,
                "validation/perplexity": perplexity,
                "validation/sequences_processed": self.processed_sequences,
                "validation/tokens_processed": (
                    self.processed_sequences * self.sequence_length
                ),
                "validation/progress_percent": 100.0,
                "validation/elapsed_seconds": self.elapsed_seconds,
                "validation/average_tokens_per_sec": (self.average_tokens_per_second),
            }
        )

    def close(self) -> None:
        if self.end_time is None:
            self.end_time = self.clock()
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
