"""Telemetry calculation and metric-schema tests."""

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from telemetry import (
    RollingRateMeter,
    TokenizationTelemetry,
    UltronTelemetry,
    ValidationTelemetry,
    format_rate,
    wandb_run_name,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeAccelerator:
    gradient_accumulation_steps = 3
    num_processes = 4
    is_main_process = False

    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, Any]]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((step, metrics))


class FakeMainAccelerator(FakeAccelerator):
    is_main_process = True

    def __init__(self) -> None:
        super().__init__()
        self.run = SimpleNamespace(id="run-123", summary={})

    def get_tracker(self, name: str, unwrap: bool = False) -> Any:
        assert name == "wandb"
        assert unwrap is True
        return self.run


def test_fresh_wandb_run_name_starts_with_timestamp(
    monkeypatch: Any,
) -> None:
    now = datetime(2026, 8, 6, 19, 33, 56, tzinfo=UTC)

    # Default pretrain and SFT names
    assert wandb_run_name("pretrain", now) == "20260806-193356-ultron-113m-pretrain"
    assert wandb_run_name("sft", now) == "20260806-193356-ultron-113m-sft"

    # Custom override
    monkeypatch.setenv("ULTRON_RUN_NAME", "custom-experiment")
    assert wandb_run_name("pretrain", now) == "20260806-193356-custom-experiment"


def test_rolling_rate_uses_recent_cumulative_samples() -> None:
    clock = FakeClock()
    meter = RollingRateMeter(window_seconds=10, clock=clock)

    assert meter.update(100).units_per_second == 0
    clock.advance(2)
    assert meter.update(140).units_per_second == pytest.approx(20)
    clock.advance(20)
    assert meter.update(240).units_per_second == pytest.approx(5)


def test_rolling_rate_rejects_invalid_window_and_resets_on_counter_rewind() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        RollingRateMeter(window_seconds=0)

    clock = FakeClock()
    meter = RollingRateMeter(window_seconds=10, clock=clock)
    meter.update(100)
    clock.advance(2)
    assert meter.update(120).units_per_second == 10
    clock.advance(1)
    assert meter.update(5).units_per_second == 0


def test_training_throughput_counts_all_workers() -> None:
    clock = FakeClock()
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator, clock=clock)

    telemetry.update_terminal_progress(4)
    clock.advance(2)
    eta = telemetry.update_terminal_progress(6)

    assert telemetry.global_tokens_per_step == 240
    assert telemetry.last_steps_per_sec == pytest.approx(1)
    assert telemetry.last_throughput == pytest.approx(240)
    assert eta == 14


def test_structured_training_metrics_have_no_legacy_duplicates() -> None:
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)
    telemetry.last_throughput = 123
    telemetry.last_steps_per_sec = 4

    telemetry.log_training_step(3, 2.5, 1e-3)

    step, metrics = accelerator.logged[0]
    assert step == 3
    assert "step" not in metrics
    assert metrics["train/loss"] == 2.5
    assert metrics["perf/tokens_per_sec"] == 123
    assert "train_loss" not in metrics
    assert "perf/eta_seconds" not in metrics
    assert "train/progress_percent" not in metrics
    assert "perf/global_tokens_per_step" not in metrics


def test_training_metrics_keep_throughput_and_dev_loss_continuous() -> None:
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)
    telemetry.last_throughput = 240
    telemetry.last_steps_per_sec = 1
    telemetry.set_last_dev_loss(2.75)

    telemetry.log_training_step(step=11, loss=3.5, lr=1e-3)
    telemetry.last_throughput = 300
    telemetry.last_steps_per_sec = 1.25
    telemetry.log_training_step(step=12, loss=3.25, lr=9e-4)

    first = accelerator.logged[0][1]
    second = accelerator.logged[1][1]
    assert first["perf/tokens_per_sec"] == 240
    assert second["perf/tokens_per_sec"] == 300
    assert first["eval/dev_loss"] == 2.75
    assert second["eval/dev_loss"] == 2.75


def test_evaluation_logs_interval_average_and_combined_chart(
    monkeypatch: Any,
) -> None:
    accelerator = FakeAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)
    chart = object()
    monkeypatch.setattr(telemetry, "_build_loss_comparison_chart", lambda: chart)

    telemetry.log_training_step(step=1, loss=4.0, lr=1e-3)
    telemetry.log_training_step(step=2, loss=2.0, lr=1e-3)
    telemetry.log_evaluation(step=3, train_loss=3.0, dev_loss=2.5, lr=1e-3)

    _, metrics = accelerator.logged[-1]
    assert "step" not in metrics
    assert metrics["train/average_loss"] == pytest.approx(3.0)
    assert metrics["eval/dev_loss"] == 2.5
    assert metrics["eval/sampled_dev_loss"] == 2.5
    assert metrics["train_vs_dev_loss"] is chart
    assert telemetry._loss_history_steps == [3]
    assert telemetry._average_train_loss_history == [pytest.approx(3.0)]
    assert telemetry._dev_loss_history == [2.5]

    telemetry.log_training_step(step=4, loss=5.0, lr=1e-3)
    telemetry.log_evaluation(step=5, train_loss=1.0, dev_loss=2.25, lr=1e-3)

    _, metrics = accelerator.logged[-1]
    assert metrics["train/average_loss"] == pytest.approx(3.0)
    assert telemetry._loss_history_steps == [3, 5]


def test_metric_definitions_use_native_wandb_step(monkeypatch: Any) -> None:
    definitions = []
    fake_wandb = SimpleNamespace(
        define_metric=lambda name, **options: definitions.append((name, options))
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    UltronTelemetry._define_wandb_metrics(FakeMainAccelerator())

    by_name = dict(definitions)
    assert "step" not in by_name
    assert by_name["train/*"] == {}
    assert by_name["eval/*"] == {}
    assert by_name["eval/sampled_dev_loss"]["hidden"] is True
    assert by_name["eval/dev_loss"]["summary"] == "min"
    assert "goal" not in by_name["eval/dev_loss"]


def test_wandb_run_id_uses_unwrapped_tracker() -> None:
    accelerator = FakeMainAccelerator()
    accelerator.run.name = "20260806-193356-ultron-113m-pretrain"
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    telemetry = UltronTelemetry(config, accelerator)

    assert telemetry.get_wandb_run_id() == "run-123"
    assert telemetry.get_wandb_run_name() == "20260806-193356-ultron-113m-pretrain"


def test_wandb_summary_contains_run_totals_and_live_results(
    monkeypatch: Any,
) -> None:
    accelerator = FakeMainAccelerator()
    config = SimpleNamespace(B=2, T=10, max_steps=20)
    UltronTelemetry._initialize_wandb_summary(config, accelerator)
    telemetry = UltronTelemetry(config, accelerator)
    monkeypatch.setattr(telemetry, "_build_loss_comparison_chart", lambda: None)
    telemetry.last_throughput = 240
    telemetry.last_steps_per_sec = 1

    telemetry.log_training_step(step=1, loss=3.5, lr=1e-3)
    telemetry.log_evaluation(step=2, train_loss=3.0, dev_loss=2.5, lr=9e-4)

    assert accelerator.run.summary["training/total_steps"] == 20
    assert accelerator.run.summary["training/total_tokens"] == 4_800
    assert accelerator.run.summary["training/current_step"] == 2
    assert accelerator.run.summary["training/tokens_processed"] == 480
    assert accelerator.run.summary["latest/train_loss"] == 3.0
    assert accelerator.run.summary["latest/dev_loss"] == 2.5
    assert accelerator.run.summary["best/dev_loss"] == 2.5
    assert accelerator.run.summary["validation/evaluations"] == 1

    telemetry.log_evaluation(step=3, train_loss=3.2, dev_loss=2.8, lr=8e-4)
    assert accelerator.run.summary["latest/dev_loss"] == 2.8
    assert accelerator.run.summary["best/dev_loss"] == 2.5
    assert accelerator.run.summary["validation/evaluations"] == 2


def test_tokenization_eta_and_validation() -> None:
    clock = FakeClock()
    telemetry = TokenizationTelemetry(
        target_tokens=1_000,
        start_tokens=100,
        clock=clock,
        enabled=False,
    )
    clock.advance(2)

    eta = telemetry.update(added_tokens=200, current_total=300)

    assert telemetry.last_tokens_per_second == pytest.approx(100)
    assert eta == 7
    with pytest.raises(ValueError):
        telemetry.update(added_tokens=-1, current_total=300)
    with pytest.raises(ValueError, match="must equal"):
        telemetry.update(added_tokens=5, current_total=310)


@pytest.mark.parametrize(
    ("target", "start"),
    [(0, 0), (-1, 0), (10, -1), (10, 11)],
)
def test_tokenization_telemetry_rejects_invalid_bounds(target: int, start: int) -> None:
    with pytest.raises(ValueError):
        TokenizationTelemetry(
            target_tokens=target,
            start_tokens=start,
            enabled=False,
        )


def test_validation_telemetry_reports_local_timing_and_throughput() -> None:
    clock = FakeClock()
    accelerator = FakeAccelerator()
    telemetry = ValidationTelemetry(
        total_sequences=10,
        sequence_length=20,
        accelerator=accelerator,
        clock=clock,
    )
    clock.advance(2)

    eta = telemetry.update(processed_sequences=4, mean_loss=2.5)
    telemetry.close()

    assert telemetry.last_tokens_per_second == pytest.approx(40)
    assert telemetry.average_tokens_per_second == pytest.approx(40)
    assert telemetry.elapsed_seconds == 2
    assert eta == 3

    with pytest.raises(ValueError, match="backwards"):
        telemetry.update(processed_sequences=3, mean_loss=2.0)


def test_validation_telemetry_logs_throttled_wandb_metrics() -> None:
    clock = FakeClock()
    accelerator = FakeMainAccelerator()
    telemetry = ValidationTelemetry(
        total_sequences=10,
        sequence_length=20,
        accelerator=accelerator,
        clock=clock,
    )
    clock.advance(2)

    telemetry.update(processed_sequences=4, mean_loss=2.5)

    step, metrics = accelerator.logged[-1]
    assert step == 4
    assert metrics["validation/loss"] == 2.5
    assert metrics["validation/tokens_per_sec"] == pytest.approx(40)
    assert metrics["validation/progress_percent"] == 40
    assert accelerator.run.summary["validation/tokens_processed"] == 80


def test_validation_telemetry_finalizes_wandb_summary() -> None:
    clock = FakeClock()
    accelerator = FakeMainAccelerator()
    telemetry = ValidationTelemetry(
        total_sequences=10,
        sequence_length=20,
        accelerator=accelerator,
        clock=clock,
    )
    clock.advance(5)
    telemetry.update(processed_sequences=10, mean_loss=2.0)
    telemetry.close()

    telemetry.finish(loss=2.0, perplexity=math.exp(2.0))

    summary = accelerator.run.summary
    assert summary["validation/status"] == "complete"
    assert summary["validation/loss"] == 2.0
    assert summary["validation/sequences_processed"] == 10
    assert summary["validation/tokens_processed"] == 200
    assert summary["validation/elapsed_seconds"] == 5
    assert summary["validation/average_tokens_per_sec"] == 40


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0, "— tok/s"),
        (-1, "— tok/s"),
        (math.nan, "— tok/s"),
        (math.inf, "— tok/s"),
        (1_500, "1.5k tok/s"),
        (2_500_000, "2.50M tok/s"),
        (3_000_000_000, "3.00G tok/s"),
    ],
)
def test_format_rate(rate: float, expected: str) -> None:
    assert format_rate(rate, "tok") == expected
    assert format_rate(rate, "tok") == expected
