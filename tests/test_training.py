"""CPU-safe training-loop and checkpoint regression tests."""

import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import UltronConfig
from train import build_config
from trainer import UltronTrainer


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, inputs, targets=None):
        logits = self.head(self.embedding(inputs))
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
            )
        return SimpleNamespace(logits=logits, loss=loss)


class FakeAccelerator:
    gradient_accumulation_steps = 1
    num_processes = 1
    sync_gradients = True
    device = torch.device("cpu")
    is_main_process = True

    def __init__(self):
        self.skipped_batches = None
        self.saved = 0
        self.loaded = 0

    def accumulate(self, _model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        return tensor

    def skip_first_batches(self, dataloader, count):
        self.skipped_batches = count
        return list(dataloader)[count:]

    def save_state(self, directory):
        self.saved += 1
        os.makedirs(directory, exist_ok=True)

    def load_state(self, _directory):
        self.loaded += 1

    def wait_for_everyone(self):
        pass

    def print(self, _message):
        pass


class FakeTelemetry:
    def __init__(self):
        self.evaluations = []

    def print_message(self, _message):
        pass

    def update_terminal_progress(self, _step, loss):
        return 0

    def log_training_step(self, **_kwargs):
        pass

    def log_evaluation(self, step, train_loss, dev_loss, lr):
        self.evaluations.append((step, train_loss, dev_loss, lr))

    def get_wandb_run_id(self):
        return None

    def close(self):
        pass


def make_trainer(max_steps=2, is_test_mode=True, num_processes=1):
    inputs = torch.randint(0, 16, (4, 6))
    targets = torch.roll(inputs, shifts=-1, dims=1)
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    config = UltronConfig(
        B=1,
        T=6,
        max_steps=max_steps,
        warmup_steps=1,
        learning_rate=1e-2,
        min_lr=1e-3,
        eval_interval=1,
        eval_batches=20,
        data_seed=1337,
    )
    config.is_test_mode = is_test_mode
    model = TinyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    accelerator = FakeAccelerator()
    accelerator.num_processes = num_processes
    trainer = UltronTrainer(
        model,
        None,
        optimizer,
        dataloader,
        dataloader,
        config,
        accelerator,
    )
    trainer.telemetry = FakeTelemetry()
    return trainer


def test_training_pipeline_reaches_evaluation_without_checkpointing():
    trainer = make_trainer()

    trainer.train()

    assert trainer.step == 2
    assert len(trainer.telemetry.evaluations) == 2
    assert trainer.accelerator.saved == 0


def test_resume_skips_consumed_batches():
    trainer = make_trainer(max_steps=2)
    trainer.step = 1

    trainer.train()

    assert trainer.accelerator.skipped_batches == 1
    assert trainer.step == 2


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, (0, 0)), (1, (0, 1)), (3, (0, 3)), (4, (1, 0)), (9, (2, 1))],
)
def test_data_position_crosses_shuffle_epochs_exactly(step, expected):
    trainer = make_trainer(max_steps=10)

    assert trainer._data_position(step) == expected

    with pytest.raises(ValueError, match="negative"):
        trainer._data_position(-1)


def test_sampled_validation_rotates_and_wraps_batches():
    trainer = make_trainer()
    trainer.config.eval_batches = 2
    expected = list(trainer.dev_loader)

    first = list(trainer._sample_dev_batches())
    second = list(trainer._sample_dev_batches())
    wrapped = list(trainer._sample_dev_batches())

    assert torch.equal(first[0][0], expected[0][0])
    assert torch.equal(first[1][0], expected[1][0])
    assert torch.equal(second[0][0], expected[2][0])
    assert torch.equal(second[1][0], expected[3][0])
    assert torch.equal(wrapped[0][0], expected[0][0])
    assert trainer.dev_batch_cursor == 2


def test_custom_test_length_remains_checkpoint_safe():
    config = build_config(SimpleNamespace(mode="test", max_steps=7))

    assert config.is_test_mode is True
    assert config.max_steps == 7


def test_learning_rate_schedule_hits_warmup_stable_and_decay_boundaries():
    trainer = make_trainer(max_steps=10)
    trainer.config.warmup_steps = 2
    trainer.config.learning_rate = 1e-2
    trainer.config.min_lr = 1e-3
    trainer.decay_start_step = 8

    trainer.step = 0
    assert trainer.update_learning_rate() == 5e-3
    trainer.step = 2
    assert trainer.update_learning_rate() == 1e-2
    trainer.step = 8
    assert trainer.update_learning_rate() == pytest.approx(1e-2)
    trainer.step = 10
    assert trainer.update_learning_rate() == pytest.approx(1e-3)


def test_total_training_tokens_come_from_runtime_configuration():
    trainer = make_trainer(max_steps=7, num_processes=3)

    assert trainer.tokens_per_step == 1 * 1 * 6 * 3
    assert trainer.total_training_tokens == 1 * 1 * 6 * 3 * 7


def test_only_main_process_writes_checkpoint_metadata(tmp_path):
    trainer = make_trainer(is_test_mode=False)
    trainer.accelerate_dir = str(tmp_path / "checkpoint")
    trainer.accelerator.is_main_process = False

    trainer.save_checkpoint()

    state_file = tmp_path / "checkpoint" / "training_state.json"
    assert not state_file.exists()

    trainer.accelerator.is_main_process = True
    trainer.step = 7
    trainer.dev_batch_cursor = 3
    trainer.save_checkpoint()

    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["step"] == 7
    assert state["max_steps"] == trainer.config.max_steps
    assert state["data_seed"] == trainer.config.data_seed
    assert state["dev_batch_cursor"] == 3
    assert state["model_config"] == trainer.config.to_metadata()


def test_checkpoint_load_restores_step_and_validation_cursor(tmp_path):
    trainer = make_trainer()
    trainer.accelerate_dir = str(tmp_path / "checkpoint")
    os.makedirs(trainer.accelerate_dir)
    state_file = tmp_path / "checkpoint" / "training_state.json"
    state_file.write_text(
        json.dumps({"step": 9, "dev_batch_cursor": 6})
    )

    trainer.load_checkpoint()

    assert trainer.accelerator.loaded == 1
    assert trainer.step == 9
    assert trainer.dev_batch_cursor == 2


def test_checkpoint_load_rejects_shuffle_seed_drift(tmp_path):
    trainer = make_trainer()
    trainer.accelerate_dir = str(tmp_path / "checkpoint")
    os.makedirs(trainer.accelerate_dir)
    state_file = tmp_path / "checkpoint" / "training_state.json"
    state_file.write_text(
        json.dumps({"step": 1, "data_seed": trainer.config.data_seed + 1})
    )

    with pytest.raises(RuntimeError, match="data_seed"):
        trainer.load_checkpoint()
