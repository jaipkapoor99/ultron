# Engineering Journey

🎵🎵🎸🎸

> “Don't stop believin'\
> Hold on to that feelin'\
> Streetlights, people”
>
> — Don't Stop Believing by Journey
> 🎵🎵🎸🎸

## Building Ultron Through Iteration

Ultron developed through repeated cycles of implementation, measurement,
failure, and correction. The project became more rigorous because its early
assumptions were tested against real training behavior rather than left
unchallenged.

## Making the Dataset Resumable

The original tokenization pipeline was vulnerable to interrupted downloads,
hardcoded settings, and uncertain resume positions. It evolved into an
exact-resume pipeline built around the streaming dataset's native cursor,
atomically committed shards, pinned revisions, and a validated pending-token
state.

The final pipeline produced 100 verified shards containing exactly 10 billion
tokens. Uploading is now blocked unless every shard and metadata pair passes
validation.

A late checkpoint resume exposed a separate process-boundary failure. Python
3.14 starts multiprocessing workers through `forkserver`; pickling a dataset
that retained open NumPy memmaps copied shard contents into RAM. The main
process and one worker consumed more than 50 GiB before the kernel OOM killer
terminated training, leaving leaked-semaphore warnings during cleanup.

The dataset now serializes only shard paths and sequence metadata. Every
DataLoader worker opens and caches its own lightweight memmap views lazily.
Forkserver and pickle-size regression tests enforce this invariant, turning an
otherwise ambiguous shutdown warning into a permanent process-safety rule.

The repaired run then completed all 152,587 optimizer steps and processed
9,999,941,632 model tokens. W&B recorded 55,083 seconds of cumulative runtime,
181,543 effective tokens per second end to end, and 189,475 tokens per second
in the final rolling window. After the worker fix, whole-system RAM averaged
16.9% and peaked at 20.6%, instead of climbing past 50 GiB. The RTX 5090
averaged about 17.0 GiB of allocated VRAM and 97.7% GPU utilization; the
26.1 GiB VRAM maximum was a transient checkpoint-save spike. Host CPU
utilization averaged 12.3% across 16 logical CPUs, confirming that training
remained GPU-bound.

The subsequent complete dev pass evaluated 499,998,720 tokens in 17 minutes
21 seconds. It measured a full-validation loss of 2.964989 and perplexity of
19.3945 at 480,436 tokens per second average. Validation remained GPU-bound
at 97.6% average GPU utilization while allocating 9.4 GiB VRAM on average,
demonstrating that the final metric can be reproduced substantially more
cheaply than training.

Qualitative generation exposed exact phrase loops that loss and multiple-choice
benchmarks do not measure. Sampling now combines a modest 1.1 repetition
penalty with a no-repeat 3-gram constraint. Model execution remains in
`UltronModel.generate`, while these adjustable decoding rules remain isolated
in `scripts/generate.py`. A final-checkpoint stress suite generated three
continuations each for AI, mathematics, and science-fiction prompts. All nine
remained grammatical and avoided phrase loops without visible
over-penalization, so the corrected policy became the documented default.

## Learning the Importance of Data Geometry

Training initially used 1,024-token windows with a stride of 256, producing 75%
overlap between adjacent samples. Shuffling obscured the cost by spreading
related windows across batches, but it did not eliminate duplicated tokens.

Removing shuffling exposed the deeper problem: sequential training repeatedly
presented nearly identical windows and traversed only about 2.5 billion unique
source tokens during a nominal 10-billion-token run.

The resulting design is stronger. Windows now use a stride equal to the context
length, eliminating overlap, while deterministic epoch-specific shuffling
preserves batch diversity. The sampler uses a fixed seed and reconstructible
epoch state, so checkpoint resume remains exact without increasing VRAM usage.

## Strengthening Validation

The historical pipeline randomly split overlapping windows. Related token
regions could therefore appear in both training and validation, producing an
optimistic validation loss.

Ultron now splits at shard boundaries, keeping training and validation tokens
separate. Frequent evaluation is explicitly labeled as a sampled estimate, and
each evaluation advances to the next dev batches instead of repeatedly scoring
the beginning of the partition. The cursor wraps deterministically and survives
checkpoint resume. A separate validation script supports a complete pass when
a definitive result is required.

## Improving Checkpoint Reliability

Resume behavior progressed from simply restoring model weights to preserving
the complete optimization state, training step, W&B identity, and deterministic
data position. Fast-forwarding now derives the correct shuffle epoch and batch
offset instead of depending on an accidental sampler order.

Checkpoint uploads include the optimizer state so training can continue
faithfully rather than merely restart from model weights.

Exact-resume assumptions are enforced rather than implied. Training rejects
incomplete gradient-accumulation groups at epoch boundaries, and checkpoint
loading rejects a changed shuffle seed that would silently alter the data
sequence.

## Replacing Legacy Components

The external Muon implementation was replaced with PyTorch's official
optimizer. Parameter partitioning is now explicit and tested: hidden
two-dimensional matrices use Muon, while embeddings, normalization parameters,
and other tensors use the appropriate AdamW groups.

The model also gained stricter checkpoint compatibility checks, clearer
parameter accounting, and safer tied-weight loading.

## Turning Telemetry Into an Engineering Tool

Telemetry began as a noisy collection of values, including graphing data such
as ETA that belonged only in the terminal. It was reorganized around useful
training signals:

- continuous throughput;
- sampled and continuous dev-loss reporting;
- interval-average training loss;
- a combined train-versus-dev chart;
- explicit W&B summaries;
- rolling terminal throughput and ETA.

This made telemetry useful for diagnosis. Matched-step comparisons between runs
revealed that a slow loss curve was a genuine optimization problem rather than
only a validation artifact.

## Building a Reproducible Project

The repository gained Python 3.14 support, a documented development workflow,
CI, broader regression tests, full validation tooling, and contributor
guidance. `pyproject.toml` is now the sole dependency manifest. Runtime and
development dependencies, including pytest and Ruff, are declared together
instead of being duplicated across hand-maintained requirements files.

Generated lock and requirements files are deliberately untracked. For CPU CI,
`uv` resolves a temporary requirements file from `pyproject.toml` while omitting
the project-selected PyTorch package, then installs the official CPU PyTorch
wheel separately. This keeps local CUDA selection flexible without allowing CI
and contributor setup to drift from the canonical dependency declarations.

Ruff was introduced as an executable quality gate rather than a passive editor
preference. Its rules cover likely bugs, invalid imports and statements,
Pyflakes errors, deterministic import ordering, unnecessary collection
construction, safe simplifications, Python 3.14 modernization, and Ruff-native
correctness checks. The initial migration used automatic safe fixes first and
reserved manual edits for the remaining cases where intent mattered.

CI now rebuilds dependencies from project metadata, verifies the installed
environment, runs Ruff, byte-compiles the source tree, and executes the
CPU-safe test suite. CUDA compilation and training remain separate hardware
checks.

Test counts alone did not show which production paths were actually exercised,
so the suite gained `pytest-cov` with branch measurement. The first complete
run established a 69.89% baseline across 1,618 statements and 468 branches. CI
enforces a 69% floor and prints missing lines, turning coverage into a
regression signal without pretending that percentage alone measures test
quality.

Dataset, telemetry, optimizer, checkpoint, and training-loop behavior are
covered by CPU-safe tests, with CUDA validation kept as a separate hardware
check.

The suite now exercises successful behavior and deliberate corruption:
invalid window geometry, shard-boundary errors,
non-deterministic resume risks, malformed tokenization state, missing or
truncated shards, incompatible checkpoints, telemetry edge cases, and unsafe
upload conditions.

## Supervised Fine-Tuning (SFT) & Instruction Alignment

Following pre-training, Ultron underwent instruction fine-tuning to transition
from raw document continuation to structured conversational assistance.

### Streaming SmolTalk Sharding & Loss Masking

The tokenization pipeline was generalized into a modular `sharding/` package.
SmolTalk conversations were formatted with ChatML (`<|im_start|>`, `<|im_end|>`)
and tokenized into 191 binary shards (955 million tokens). To enforce assistant
learning without penalizing the model for user prompts or system headers, target
arrays masked all non-assistant tokens with `-1` (`int32`), while input tokens
were stored as compact `uint16`.

### The Autoregressive Target Shift Bug

An early SFT run achieved a suspiciously perfect validation loss of `0.000084`
and perplexity of `1.0001`. A deep-dive into `sft_dataset.py` revealed that both
inputs and targets were sliced without offset (`inp = tokens[i:i+T]`,
`tgt = targets[i:i+T]`). Because the causal self-attention mechanism at position
$t$ already contained token $x_t$ in its receptive field, predicting position
$t$ was a trivial identity copy of the current input token rather than a next-token
prediction.

The dataloader was immediately corrected to enforce strict autoregressive
alignment (`tgt = targets[i+1:i+T+1]`), and a dedicated contract test was added
to `tests/test_sft.py`. The subsequent verified SFT run converged cleanly from an
initial loss of ~2.2 down to a genuine dev loss of `1.4662` (perplexity `4.3328`)
across 43,938 held-out dev sequences.

### Hugging Face Hub Monolithic Upload Stalls

When publishing large training checkpoints to brand-new, empty repositories,
`huggingface_hub`'s `upload_folder` bundled 1.06 GB of binary weights into a
single commit payload, stalling the server-side Git tree creation. Replacing
the monolithic call with sequential `upload_file` transactions ensured that every
tensor (`model.safetensors`, `optimizer.bin`, `optimizer_1.bin`) is committed
independently with immediate progress tracking and automatic LFS registration.

### The Reality of 113M Parameter Models

Fine-tuning a 113M model on instruction datasets presents a fascinating look into
parameter capacity constraints:

1. **Conversational Mechanics:** The model flawlessly internalizes ChatML syntax,
   assistant greetings, multi-turn state, and turn termination (`<|im_end|>`).
1. **The "Alignment Tax":** Standard zero-shot multiple-choice benchmarks
   exhibit a slight distribution shift (Macro-average: 40.41% base vs 38.58%
   instruct), while semantic disambiguation tasks improved (Winogrande: 49.17%
   to 50.83%).
1. **Parametric Capacity Bounds:** While outputs from a 113M model may seem
   insufficient or prone to entity hallucinations compared to multi-billion
   parameter cloud models, this is the honest reality of the sub-200M parameter
   regime on a 10B token budget. Ultron-113M establishes clean linguistic syntax,
   solid commonsense reasoning, and disciplined ChatML turn-taking in a lightweight
   footprint that executes locally in milliseconds with under 250MB VRAM.

## Additional Engineering Lessons

### Accelerate Setup and Launcher Protocols

- **Strict launcher enforcement:** Early execution through `python3` failed
  because distributed process groups were not initialized. Standardizing on
  `accelerate launch` across every entry point made device setup explicit.
- **DeepSpeed and dual optimizers:** DeepSpeed's unified optimizer engine
  conflicted with Muon for hidden matrices and AdamW for embeddings and
  normalization parameters. Native PyTorch BF16 and `torch.compile` retained
  the intended optimizer split while reaching approximately 186.3k tokens/s.

### Python Environment and C Headers

- **The `Python.h` bottleneck:** The system Python 3.14 installation lacked
  development headers, causing `torch.compile` to fail. An uv-managed Python
  3.14.6 environment supplied the standalone headers required by the compiler.
- **Built-in Muon:** PyTorch 2.13 provides `torch.optim.Muon`, eliminating the
  external optimizer dependency and its legacy checkpoint contract.

### High-Throughput Tokenization and Memory Mapping

- **Rust batch tokenization:** Replacing Python tokenization loops with
  `backend_tokenizer.encode_batch` increased throughput from roughly 40k to
  4.34 million tokens/s.
- **Bounded sample reads:** Contiguous 100-million-token `uint16` shards support
  memory-mapped access while allocating only the requested window and its
  `int64` conversion.
- **Non-overlapping shuffled windows:** A 1,024-token stride removes adjacent
  overlap, while seeded epoch permutations retain batch diversity and exact
  resume behavior.
- **Leakage-safe validation:** Shard-boundary partitioning keeps related token
  windows out of opposing train and validation splits.
- **Rotating sampled validation:** Frequent validation advances through the dev
  loader, wraps deterministically, and persists its cursor in checkpoints.

## Lessons Carried Forward

1. Measure unique corpus coverage, not only nominal processed tokens.
1. Treat stride, sampling, shuffling, and resume behavior as one system.
1. Compare averaged training loss at matched steps before blaming validation.
1. Validate all input artifacts before starting an expensive run.
1. Make important state explicit, versioned, and testable.
1. Prefer reproducible behavior over behavior that merely appears deterministic.
1. Treat failed runs as evidence that improves the next design.
1. Test corruption and resume boundaries, not only successful execution.
1. Keep dependency declarations singular; generate environment-specific inputs
   at the boundary that needs them.
1. Run safe automated lint fixes before making narrow semantic corrections.
1. In autoregressive SFT, always verify the $+1$ target index offset to avoid
   trivial identity-copy leakage.
1. Acknowledge model capacity scaling bounds: evaluate small models on formatting
   and turn discipline rather than deep encyclopedic trivia recall.

Ultron's engineering journey is not a story of avoiding mistakes. It is a story
of converting each mistake into a stronger invariant, a clearer test, and a
more dependable training system.
