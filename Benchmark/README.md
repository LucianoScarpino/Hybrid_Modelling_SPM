# Reproducible benchmark suite

This folder runs the project models through one measurement protocol and
collects machine-readable results. Every seed/repeat is executed in a separate
Python process so that imports, device memory, checkpoints, and failures from
one model do not contaminate another run.

The suite does not call plotting code and does not overwrite the scientific
datasets. Temporary checkpoints created by the legacy PINN training routines
remain inside the corresponding run folder.

## Models

- `reference_fvm`: adaptive-diffusivity thermo-diffusive finite-volume model;
- `simplified_fvm`: constant-diffusivity finite-volume model;
- `fvm_hybrid`: simplified FVM plus radial-profile FNN correction;
- `pinn`: standalone simplified-physics PINN;
- `hybrid_pinn`: frozen PINN plus post-training residual FNN;
- `thermal_sindyc`: sparse thermal equation identification;
- `thermal_ffn`: neural thermal equation and rollout.

The FVM--FNN worker also evaluates every available
`Hybrid_Model/Dataset/Variants/simplified_simulation_dataset_*.csv` with the
same trained network. The resulting baseline/hybrid errors and RMSE reduction
are stored under `robustness.variant_*`; no retraining is performed for those
tests.

## First check

Run a deliberately short end-to-end smoke test:

```bash
cd Benchmark
python run_benchmarks.py --preset smoke
```

The smoke preset uses one seed, very few training epochs, and one or two timing
samples. Its purpose is to validate the pipeline, not to produce reportable
accuracy or timing conclusions.

## Standard and publication runs

```bash
# Three independent initialization seeds and production training settings.
python run_benchmarks.py --preset standard

# Five seeds, two fresh-process repeats, and expanded timing repetitions.
python run_benchmarks.py --preset publication
```

With all seven models selected, `standard` runs seeds `11`, `26`, and `42`
once, for 21 isolated executions. `publication` runs seeds `7`, `19`, `26`,
`41`, and `73` twice, for 70 isolated executions. A repeat uses the same
model seed in a fresh process; it checks determinism and machine-time
variability but does not constitute an additional independent initialization.
The finalized publication run `20260903_085820` completed all 70 executions
without failures or skipped models.

The standard preset uses a lower-overhead PINN protocol: validation is run
every 10 epochs, detailed diagnostics every 100 epochs, materially small
validation changes do not reset early stopping, and larger minibatches improve
accelerator utilization. Its PINN L-BFGS refinement is capped at 100
iterations; the publication preset retains the 500-iteration cap. These values
are stored in `effective_config.json` and the actual epochs completed are
included in each run result.

The FVM--FNN correction model defaults to CPU even when the global device is
`auto`. Its dataset and network are small, so on Apple Silicon the MPS dispatch
overhead can dominate the computation. Set `fvm_hybrid.device` in a custom
configuration if another backend is demonstrably faster on the target machine.

Both presets can be restricted to selected models:

```bash
python run_benchmarks.py \
  --preset standard \
  --models reference_fvm simplified_fvm fvm_hybrid \
  --seeds 11 26 42 57 101
```

Training the full PINN for several seeds is expensive. Run the finite-volume
and hybrid-FVM group first, inspect its outputs, and schedule PINN runs
separately when necessary:

```bash
python run_benchmarks.py --preset standard --models pinn hybrid_pinn
```

Use `--timeout-seconds` to impose a per-run limit and `--fail-fast` when a
failed run should stop the suite. Runs are intentionally sequential: concurrent
GPU/MPS workloads would invalidate latency and training-time comparisons.

If a suite is interrupted, resume it with its timestamped result directory:

```bash
python run_benchmarks.py --resume results/YOUR_RUN_DIRECTORY
```

The saved effective configuration and model/seed selection are reused.
Successful `result.json` files are retained and only incomplete or failed runs
are executed again. Do not combine `--resume` with `--config`, `--models`,
`--seeds`, `--repeats`, or `--output`. A run created before the optimized PINN
configuration was introduced resumes with its original settings; start a new
standard run to use the optimized protocol.

PySINDy is optional. If it is unavailable, `thermal_sindyc` is recorded in
`skipped.csv` while the remaining models continue. Pass `--strict-dependencies`
to turn this condition into a failed run.

Human engineering effort cannot be inferred reliably from runtime or Git
timestamps. Copy `effort_log_template.csv`, fill measured person-hours, and
include it explicitly:

```bash
python run_benchmarks.py --preset standard --effort-log effort_log.csv
```

The suite then produces `effort_summary.csv` in person-hours and 8-hour
person-days and records the source log hash in the manifest.

## Configuration overrides

`default_config.json` contains all numerical and training settings. A small
override file is enough to change an experiment:

```json
{
  "device": "cpu",
  "timing": {
    "inference_repetitions": 50
  },
  "fvm_hybrid": {
    "epochs": 5000
  },
  "hybrid_pinn": {
    "physics_diagnostics": true
  }
}
```

Apply it with:

```bash
python run_benchmarks.py --preset standard --config my_overrides.json
```

User overrides are applied after the selected preset. The effective merged
configuration is copied into the result folder.

## Result layout

Each invocation creates `Benchmark/results/YYYYMMDD_HHMMSS/` containing:

- `raw_runs.csv`: one flattened row for each successful seed/repeat;
- `summary.csv`: one row per model and numeric metric, with count, mean,
  sample standard deviation, median, minimum, maximum, p05, p95, and a 95%
  normal-approximation interval;
- `summary_wide.csv`: a compact subset of the same statistics;
- `model_comparison.csv`: comparable online latency, speedup relative to the
  reference FVM, and a compute-only break-even number of simulation calls;
- `robustness_comparison.csv`: frozen FVM--FNN performance for every available
  simplified-model variant;
- `REPORT.md`: numerical comparison table and interpretation notes;
- `manifest.json`: exact configuration, Git revision and dirty state, Python
  and device information, plus SHA-256 hashes of the input datasets;
- `failures.csv`: failed runs, when present;
- `skipped.csv`: runs omitted because an optional dependency is unavailable;
- `effort_summary.csv`: optional phase-level human effort totals;
- `runs/<model>__seed_<seed>__repeat_<n>/`: JSON result and complete log for
  every isolated worker.

An accuracy--latency scatter plot is generated only when `--plot` is passed.
Numerical files remain the primary output.

## Meaning of the accuracy metrics

Concentration-field RMSE and relative L2 error compare each approximate
pipeline with the stored reference-FVM field on the same evaluation grid.
They quantify agreement with that numerical reference, not agreement with
experimental ground truth or the unknown physical solution.

The benchmark also executes the reference pipeline to measure its latency,
mass balance, and reproducibility. Its computed field error of order `1e-14`
is a self-comparison and only reflects floating-point round-off and
data-storage precision. It has no independent accuracy meaning and must not
be reported as evidence that the reference model is physically accurate.
When the reference point appears in the optional logarithmic
accuracy--latency plot, use it only as a latency anchor; comparisons of field
accuracy apply to the approximate models.

## Meaning of the timings

All timings use a monotonic wall clock. CUDA and Apple MPS are synchronized
immediately before and after measured calls.

- `training_seconds`: model fitting only, plus unavoidable checkpoint handling
  in legacy routines;
- `online_inference`: a complete requested output grid for coordinate models;
- `online_correction`: FNN correction of all profiles, excluding the physical
  baseline;
- `online_end_to_end`: simplified finite-volume simulation plus FNN correction;
- `rollout`: integration of a thermal model over a complete test trajectory;
- `worker_wall_seconds`: full process time, including imports, data loading,
  setup, training, evaluation, and result serialization;
- `offline_dataset_generation_seconds`: FVM solve plus in-memory dataset
  assembly; CSV serialization is deliberately excluded.

The correction-only FVM--FNN latency must not be presented as the complete
simulator latency. Use `online_end_to_end` for deployment comparisons.

Compute-only break-even is obtained by dividing offline model cost by the
mean online-time saving relative to the reference FVM. It excludes human
effort, reference-data generation, validation, and maintenance. For
Hybrid-PINN it also excludes production of the frozen PINN checkpoint, so
that value is an incremental correction-stage estimate rather than a full
lifecycle break-even.

Timing values are specific to the recorded implementation and hardware.
In the finalized publication run, FVM-based pipelines ran on CPU and
PINN-based pipelines on Apple MPS. Their end-to-end latencies can be compared
as deployed pipelines on that machine, but the measurements are not a
hardware-neutral ranking of the underlying algorithms.

## Seeds and repeats

`seed` controls neural initialization, minibatch ordering, and Sobol boundary
sampling. `split_seed` is separate and fixed by default, so model variability
can be measured on the same train/validation/test partition. `repeats` reruns
the same seed in a fresh process and is useful for checking determinism and
machine-time variability.

PINN progress is written to the run-specific `worker.log`. Each validation line
reports elapsed time and an upper-bound ETA based on the configured maximum
number of epochs; early stopping can make the actual duration shorter.

For scientific reporting, use at least five independent model seeds. The
summary files aggregate completed run values, so two repeats of one seed
appear as two observations even though they share the same initialization
seed. Publication results must therefore describe the normal-approximation
intervals as descriptive. For inference specifically about initialization
variability, aggregate repeats within each seed before computing uncertainty
or use a hierarchical analysis that separates seed and repeat effects.

## Important interpretation limits

The current datasets mainly support interpolation within the available
operating trajectories. Repeating seeds quantifies optimization variability;
it does not establish extrapolative robustness. New current histories, initial
conditions, diffusivities, temperatures, meshes, and noisy datasets must be
represented as separate scenario datasets and evaluated with trajectory-wise
splits before making production generalization claims.
