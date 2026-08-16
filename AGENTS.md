# Repository Guidelines

## Project Structure & Module Organization

Ultron is a PyTorch decoder-only language-model project. Core architecture and configuration live in `model.py` and `config.py`. Training is split across `train.py`, `trainer.py`, `dataset.py`, and `telemetry.py`. Operational entry points belong in `scripts/`, including tokenization, generation, evaluation, and Hugging Face uploads. Tests live in `tests/`.

Large runtime artifacts are intentionally untracked: `shards_edu/`, `accelerate_checkpoint/`, `wandb/`, and `logs/`. Do not add model weights, token shards, credentials, or generated telemetry to commits.

## Build, Test, and Development Commands

```bash
uv venv --python 3.14.6 .venv
source .venv/bin/activate
uv pip install torch==2.13.0
uv pip install -r pyproject.toml --group dev
uv run --no-sync ruff check .
pytest -q
ULTRON_TEST_COMPILE=1 pytest -q tests/test_model.py -k torch_compile
accelerate launch train.py --mode=test
accelerate launch scripts/generate.py --prompt "Hello" --samples 4
```

`pyproject.toml` is the canonical dependency source. Generated dependency files
such as `uv.lock` and `requirements.lock` are intentionally untracked. Install
the appropriate PyTorch 2.13 wheel before the project dependencies. The standard
test suite is CPU-safe. The compiler test is opt-in because it is slower and
toolchain-dependent. `train.py --mode=test` exercises the training pipeline and
requires prepared dataset shards; full training should run on a CUDA device with
BF16 support.

## Training & Data Invariants

Training windows use `step=config.T`: never restore the former 256-token stride, which duplicated 75% of adjacent 1,024-token samples. Training uses `EpochRandomSampler` with `config.data_seed`; validation remains sequential and split at shard boundaries. Resume must reconstruct the shuffle epoch and batch offset, reject seed drift, and preserve `dev_batch_cursor`. Sampled validation advances through the dev loader and wraps only after exhausting it.

Dataset instances must remain cheap under Python 3.14 `forkserver` serialization. Store shard paths and compact metadata in pickled state; open `np.memmap` objects lazily inside each process and never serialize them to DataLoader workers.

Keep W&B on its native step axis. Throughput and held dev loss are continuous; train/dev comparison uses interval-average train loss. ETA belongs in the terminal, while bookkeeping belongs in the run summary rather than separate charts. Full validation must create a separate timestamped W&B run and retain its local JSON result.

## Coding Style & Naming Conventions

Use four-space indentation and conventional Python naming: `snake_case` for functions and variables, `PascalCase` for classes, and descriptive lowercase module names. Keep model math, training control, dataset I/O, and telemetry separated. Centralize hyperparameters and remote repository identifiers in `UltronConfig`; avoid hardcoded duplicates in scripts. Prefer established PyTorch, Accelerate, Transformers, and lm-eval APIs over custom infrastructure.

No formatter is currently enforced. Keep imports minimal, add type hints to public interfaces, and run `git diff --check` before submitting.

## Testing Guidelines

Tests use `pytest` and follow `test_<behavior>` naming. The CPU-safe suite currently has 125 tests covering model contracts, data geometry, shuffled resume, rotating validation, telemetry, atomic tokenization, shard integrity, evaluation harness behavior, multiprocessing-safe memmaps, and upload guards. Every bug fix needs a regression test; test corruption and boundary conditions as well as successful execution. Avoid downloads and full-model allocation unless explicitly marked slow.

## Commit & Pull Request Guidelines

History follows Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, and `refactor:`. Keep commits focused and imperative.

Pull requests should explain motivation, implementation, verification commands, and checkpoint or dataset compatibility impact. Include matched-step averaged losses for convergence claims and screenshots only for visual telemetry changes.

## Security & Configuration

Pass Hugging Face credentials through `HF_TOKEN`; never commit tokens. `scripts/upload_checkpoint.py` intentionally publishes the complete Accelerate checkpoint, including optimizer and RNG state, so consumers must treat the repository as a trusted training-resume artifact.
