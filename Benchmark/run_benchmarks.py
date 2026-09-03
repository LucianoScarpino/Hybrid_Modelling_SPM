#!/usr/bin/env python3
"""Run isolated multi-seed benchmarks and aggregate their statistics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    PROJECT_ROOT,
    deep_merge,
    environment_metadata,
    file_sha256,
    flatten_dict,
    git_metadata,
    load_json,
    write_json,
)
from worker import MODEL_NAMES


def effective_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], list[int], int]:
    default_path = Path(__file__).with_name("default_config.json")
    defaults = load_json(default_path)
    user_config = load_json(args.config) if args.config else {}
    catalogue = deep_merge(defaults, user_config)

    if args.preset not in catalogue["presets"]:
        choices = ", ".join(sorted(catalogue["presets"]))
        raise ValueError(f"Unknown preset {args.preset!r}. Available presets: {choices}")
    preset = catalogue["presets"][args.preset]

    base = {key: value for key, value in defaults.items() if key != "presets"}
    config = deep_merge(base, preset.get("overrides", {}))
    config = deep_merge(
        config,
        {key: value for key, value in user_config.items() if key != "presets"},
    )
    models = list(args.models or preset["models"])
    if "all" in models:
        models = list(MODEL_NAMES)
    invalid = sorted(set(models) - set(MODEL_NAMES))
    if invalid:
        raise ValueError(f"Unknown models: {', '.join(invalid)}")
    seeds = list(args.seeds or preset["seeds"])
    repeats = int(args.repeats if args.repeats is not None else preset["repeats"])
    if not seeds:
        raise ValueError("At least one seed is required.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    return config, models, seeds, repeats


def dataset_manifest(root: Path) -> list[dict[str, Any]]:
    candidates = [
        root / "Single_Particle_Model_V3" / "SPMDataset" / "full_simulation_dataset.csv",
        root / "Single_Particle_Model_V3" / "SPMDataset" / "simplified_simulation_dataset.csv",
        root / "PINN" / "Data" / "pinn_training_dataset.csv",
        root / "Hybrid_Model" / "Dataset" / "full_simulation_dataset.csv",
        root / "Hybrid_Model" / "Dataset" / "simplified_simulation_dataset.csv",
    ]
    candidates.extend(
        root
        / "Hybrid_Model"
        / "Dataset"
        / "ForThermalVariants"
        / f"full_simulation_dataset_{index}.csv"
        for index in range(8)
    )
    records = []
    for path in candidates:
        if path.exists():
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        # A resumed run may have fixed every previous failure/skip.  Remove the
        # obsolete table instead of leaving misleading data from the old state.
        path.unlink(missing_ok=True)
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_numeric_metric(key: str, value: Any) -> bool:
    if key in {"seed", "repeat_index"} or isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = [flatten_dict(result) for result in results if result.get("status") == "success"]
    rows: list[dict[str, Any]] = []
    for model_name in sorted({row["model_name"] for row in flattened}):
        model_rows = [row for row in flattened if row["model_name"] == model_name]
        metric_names = sorted(
            {
                key
                for row in model_rows
                for key, value in row.items()
                if is_numeric_metric(key, value)
            }
        )
        for metric_name in metric_names:
            values = np.asarray(
                [
                    float(row[metric_name])
                    for row in model_rows
                    if metric_name in row and is_numeric_metric(metric_name, row[metric_name])
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            standard_error = standard_deviation / np.sqrt(values.size)
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric_name,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": standard_deviation,
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "p05": float(np.percentile(values, 5)),
                    "p95": float(np.percentile(values, 95)),
                    "ci95_low": float(np.mean(values) - 1.96 * standard_error),
                    "ci95_high": float(np.mean(values) + 1.96 * standard_error),
                }
            )
    return rows


def build_wide_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wide: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        metric = row["metric"]
        if metric.startswith("worker_wall_seconds") or metric.endswith(".n"):
            continue
        key = f"{row['model_name']}::{metric}"
        wide[key] = {
            "model_name": row["model_name"],
            "metric": metric,
            "n": row["n"],
            "mean": row["mean"],
            "std": row["std"],
            "median": row["median"],
            "p95": row["p95"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
        }
    return list(wide.values())


def summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["model_name"], row["metric"]): row for row in summary_rows}


def build_model_comparison(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an operational view and a compute-only break-even estimate."""
    lookup = summary_lookup(summary_rows)

    def value(model: str, metric: str) -> float | None:
        row = lookup.get((model, metric))
        return float(row["mean"]) if row else None

    def latency_metrics(model: str) -> tuple[float | None, float | None, str | None]:
        candidates = (
            "timing.online_end_to_end",
            "timing.online_inference",
            "timing.rollout.interpolation",
            "timing.online_correction",
        )
        for prefix in candidates:
            mean_value = value(model, f"{prefix}.mean_seconds")
            if mean_value is not None:
                return mean_value, value(model, f"{prefix}.p95_seconds"), prefix
        return None, None, None

    reference_latency, _, _ = latency_metrics("reference_fvm")
    model_names = sorted({row["model_name"] for row in summary_rows})
    rows = []
    for model in model_names:
        latency, latency_p95, latency_definition = latency_metrics(model)
        training = value(model, "timing.training_seconds") or 0.0
        speedup = None
        break_even = None
        if (
            reference_latency is not None
            and latency is not None
            and not model.startswith("thermal_")
        ):
            speedup = reference_latency / latency if latency > 0.0 else None
            saving_per_call = reference_latency - latency
            if saving_per_call > 0.0:
                break_even = training / saving_per_call
        rows.append(
            {
                "model_name": model,
                "latency_definition": latency_definition,
                "mean_online_latency_seconds": latency,
                "mean_online_latency_p95_seconds": latency_p95,
                "mean_training_seconds": training,
                "speedup_vs_reference_fvm": speedup,
                "compute_only_break_even_calls_vs_reference": break_even,
                "field_relative_l2_mean": value(model, "accuracy.field.relative_l2"),
                "field_rmse_mean": value(model, "accuracy.field.rmse"),
                "mass_balance_max_abs_mean": value(
                    model, "physics.mass_balance_max_abs"
                ),
            }
        )
    return rows


def build_robustness_comparison(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = summary_lookup(summary_rows)
    prefix = "robustness."
    suffix = ".field_rmse_reduction_percent"
    variants = sorted(
        {
            row["metric"][len(prefix) : -len(suffix)]
            for row in summary_rows
            if row["model_name"] == "fvm_hybrid"
            and row["metric"].startswith(prefix)
            and row["metric"].endswith(suffix)
        }
    )

    def statistic(metric: str, name: str = "mean") -> float | None:
        row = lookup.get(("fvm_hybrid", metric))
        return float(row[name]) if row else None

    rows = []
    for variant in variants:
        variant_prefix = f"robustness.{variant}"
        rows.append(
            {
                "variant": variant,
                "initial_simplified_concentration": statistic(
                    f"{variant_prefix}.initial_simplified_concentration"
                ),
                "dimensionless_final_time": statistic(
                    f"{variant_prefix}.dimensionless_final_time"
                ),
                "baseline_field_rmse_mean": statistic(
                    f"{variant_prefix}.baseline_accuracy.field.rmse"
                ),
                "hybrid_field_rmse_mean": statistic(
                    f"{variant_prefix}.hybrid_accuracy.field.rmse"
                ),
                "rmse_reduction_percent_mean": statistic(
                    f"{variant_prefix}.field_rmse_reduction_percent"
                ),
                "rmse_reduction_percent_std": statistic(
                    f"{variant_prefix}.field_rmse_reduction_percent", "std"
                ),
                "physical_bounds_violations_mean": statistic(
                    f"{variant_prefix}.physics_against_reference_inventory."
                    "physical_bounds_violations"
                ),
            }
        )
    return rows


def format_scientific(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4e}"


def generate_markdown_report(
    path: Path,
    results: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lookup = summary_lookup(summary_rows)

    def mean(model: str, *metrics: str) -> float | None:
        for metric in metrics:
            row = lookup.get((model, metric))
            if row:
                return float(row["mean"])
        return None

    models = sorted({result["model_name"] for result in results})
    comparison = {row["model_name"]: row for row in comparison_rows}
    lines = [
        "# Benchmark summary",
        "",
        f"Generated: {manifest['created_at']}",
        "",
        "All timings are wall-clock measurements. Accelerator operations are synchronized. "
        "The end-to-end FVM--FNN timing includes the simplified finite-volume solver; "
        "the correction-only timing does not.",
        "",
        "| Model | Runs | Training [s] | Core inference [s] | End-to-end [s] | Speedup vs reference | Compute break-even [calls] | Field relative L2 | Field RMSE | Max mass residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        run_count = sum(result["model_name"] == model for result in results)
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    str(run_count),
                    format_scientific(mean(model, "timing.training_seconds")),
                    format_scientific(
                        mean(
                            model,
                            "timing.online_inference.mean_seconds",
                            "timing.online_correction.mean_seconds",
                            "timing.rollout.interpolation.mean_seconds",
                        )
                    ),
                    format_scientific(mean(model, "timing.online_end_to_end.mean_seconds")),
                    format_scientific(
                        comparison.get(model, {}).get("speedup_vs_reference_fvm")
                    ),
                    format_scientific(
                        comparison.get(model, {}).get(
                            "compute_only_break_even_calls_vs_reference"
                        )
                    ),
                    format_scientific(mean(model, "accuracy.field.relative_l2")),
                    format_scientific(mean(model, "accuracy.field.rmse")),
                    format_scientific(mean(model, "physics.mass_balance_max_abs")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Statistical outputs",
            "",
            "- `raw_runs.csv`: one row per completed seed/repeat.",
            "- `summary.csv`: long-format statistics for every numeric metric.",
            "- `summary_wide.csv`: compact mean/std/median/p95/CI95 view.",
            "- `model_comparison.csv`: operational latency, reference speedup, and compute-only break-even.",
            "- `manifest.json`: code, environment, configuration, and dataset hashes.",
            "- `runs/*/worker.log`: complete output of each isolated run.",
            "",
            "The reported 95% interval is the normal approximation "
            "mean +/- 1.96 standard errors. With fewer than five independent seeds, "
            "it should be treated as descriptive rather than inferential.",
            "",
            "The break-even excludes human development effort and reference-data generation. "
            "For Hybrid-PINN it also excludes the original frozen PINN training, so it is an "
            "incremental rather than lifecycle break-even.",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(
                f"- `{failure.get('model_name')}` seed {failure.get('seed')}: "
                f"{failure.get('error_type', 'Error')} - {failure.get('error', 'see log')}"
            )
    if robustness_rows:
        lines.extend(
            [
                "",
                "## Frozen FVM--FNN robustness",
                "",
                "| Variant | Simplified RMSE | Hybrid RMSE | RMSE reduction [%] |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in robustness_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["variant"]),
                        format_scientific(row["baseline_field_rmse_mean"]),
                        format_scientific(row["hybrid_field_rmse_mean"]),
                        format_scientific(row["rmse_reduction_percent_mean"]),
                    ]
                )
                + " |"
            )
    if skipped:
        lines.extend(["", "## Skipped runs", ""])
        for item in skipped:
            lines.append(
                f"- `{item.get('model_name')}` seed {item.get('seed')}: "
                f"{item.get('reason', 'optional dependency unavailable')}"
            )
    if manifest.get("effort_log"):
        effort = manifest["effort_log"]
        lines.extend(
            [
                "",
                "## Recorded engineering effort",
                "",
                f"Total recorded effort: {effort['recorded_person_hours']:.2f} person-hours. ",
                "See `effort_summary.csv` for the phase breakdown.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_optional_plot(output_directory: Path, summary_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    lookup = summary_lookup(summary_rows)
    points = []
    for model in MODEL_NAMES:
        error_row = lookup.get((model, "accuracy.field.relative_l2"))
        timing_row = (
            lookup.get((model, "timing.online_end_to_end.mean_seconds"))
            or lookup.get((model, "timing.online_inference.mean_seconds"))
            or lookup.get((model, "timing.online_correction.mean_seconds"))
        )
        if error_row and timing_row and error_row["mean"] > 0 and timing_row["mean"] > 0:
            points.append((model, timing_row["mean"], error_row["mean"]))
    if not points:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    for model, timing, error in points:
        axis.scatter(timing, error, s=55)
        axis.annotate(model, (timing, error), xytext=(5, 5), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Mean online wall time [s]")
    axis.set_ylabel("Mean field relative L2 error")
    axis.set_title("Accuracy--latency trade-off")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "accuracy_latency.png", dpi=200)
    plt.close(figure)


def summarize_effort(path: Path) -> list[dict[str, Any]]:
    """Aggregate a manually maintained effort log without inventing work hours."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    required = {"phase", "person_hours"}
    if not records:
        return []
    missing = required - set(records[0])
    if missing:
        raise ValueError(
            f"Effort log {path} is missing columns: {', '.join(sorted(missing))}"
        )
    totals: dict[str, float] = {}
    for record in records:
        raw_hours = (record.get("person_hours") or "").strip()
        if not raw_hours:
            continue
        phase = (record.get("phase") or "unclassified").strip() or "unclassified"
        totals[phase] = totals.get(phase, 0.0) + float(raw_hours)
    rows = [
        {"phase": phase, "person_hours": hours, "person_days_at_8h": hours / 8.0}
        for phase, hours in sorted(totals.items())
    ]
    total_hours = sum(totals.values())
    if rows:
        rows.append(
            {
                "phase": "TOTAL",
                "person_hours": total_hours,
                "person_days_at_8h": total_hours / 8.0,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "standard", "publication"))
    parser.add_argument("--models", nargs="+", choices=(*MODEL_NAMES, "all"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--config", type=Path, help="JSON overrides applied after the preset.")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Resume an existing result directory using its saved configuration; "
            "successful runs are reused."
        ),
    )
    parser.add_argument(
        "--effort-log",
        type=Path,
        help="Optional CSV based on effort_log_template.csv.",
    )
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--strict-dependencies",
        action="store_true",
        help="Treat a missing optional PySINDy installation as a failed run.",
    )
    parser.add_argument("--plot", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = PROJECT_ROOT
    if args.resume and args.output:
        raise ValueError("--resume and --output cannot be used together.")
    if args.resume and any(
        value is not None
        for value in (args.config, args.models, args.seeds, args.repeats)
    ):
        raise ValueError(
            "--resume uses the saved configuration, models, seeds, and repeats; "
            "do not combine it with selection or configuration overrides."
        )

    if args.resume:
        output_directory = args.resume.resolve()
        if not output_directory.is_dir():
            raise FileNotFoundError(output_directory)
        effective_config_path = output_directory / "effective_config.json"
        manifest_path = output_directory / "manifest.json"
        if not effective_config_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                "A resumable directory must contain effective_config.json and manifest.json."
            )
        config = load_json(effective_config_path)
        manifest = load_json(manifest_path)
        models = list(manifest["models"])
        seeds = [int(seed) for seed in manifest["seeds"]]
        repeats = int(manifest["repeats"])
        manifest.setdefault("resume_events", []).append(
            {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "environment": environment_metadata(),
                "git": git_metadata(root),
            }
        )
    else:
        config, models, seeds, repeats = effective_configuration(args)
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = (
            args.output.resolve()
            if args.output
            else (Path(__file__).resolve().parent / "results" / run_stamp)
        )
        output_directory.mkdir(parents=True, exist_ok=False)
        effective_config_path = output_directory / "effective_config.json"
        write_json(effective_config_path, config)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(root),
            "preset": args.preset,
            "models": models,
            "seeds": seeds,
            "repeats": repeats,
            "configuration": config,
            "environment": environment_metadata(),
            "git": git_metadata(root),
            "datasets": dataset_manifest(root),
        }
    effort_rows: list[dict[str, Any]] = []
    if args.effort_log:
        effort_path = args.effort_log.resolve()
        effort_rows = summarize_effort(effort_path)
        write_csv(output_directory / "effort_summary.csv", effort_rows)
        manifest["effort_log"] = {
            "path": str(effort_path),
            "sha256": file_sha256(effort_path),
            "recorded_person_hours": (
                effort_rows[-1]["person_hours"] if effort_rows else 0.0
            ),
        }
    manifest["status"] = "running"
    manifest["last_started_at"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("completed_at", None)
    manifest.pop("interrupted_at", None)
    write_json(output_directory / "manifest.json", manifest)

    worker_path = Path(__file__).with_name("worker.py")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    interrupted = False
    interrupted_run: str | None = None
    total = len(models) * len(seeds) * repeats
    current = 0
    for model_name in models:
        for seed in seeds:
            for repeat_index in range(repeats):
                current += 1
                run_name = f"{model_name}__seed_{seed}__repeat_{repeat_index + 1}"
                run_directory = output_directory / "runs" / run_name
                run_directory.mkdir(parents=True, exist_ok=bool(args.resume))
                result_path = run_directory / "result.json"
                log_path = run_directory / "worker.log"
                if args.resume and result_path.exists():
                    existing_result = load_json(result_path)
                    if existing_result.get("status") == "success":
                        results.append(existing_result)
                        print(
                            f"[{current}/{total}] {model_name} | seed={seed} | "
                            f"repeat={repeat_index + 1} | REUSED",
                            flush=True,
                        )
                        continue
                if (
                    model_name == "thermal_sindyc"
                    and importlib.util.find_spec("pysindy") is None
                    and not args.strict_dependencies
                ):
                    result = {
                        "status": "skipped",
                        "model_name": model_name,
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "reason": (
                            "Optional dependency 'pysindy' is not installed in "
                            f"{sys.executable}."
                        ),
                    }
                    write_json(result_path, result)
                    log_path.write_text(result["reason"] + "\n", encoding="utf-8")
                    skipped.append(result)
                    print(
                        f"[{current}/{total}] {model_name} | seed={seed} | "
                        f"repeat={repeat_index + 1} | SKIPPED",
                        flush=True,
                    )
                    continue
                command = [
                    sys.executable,
                    str(worker_path),
                    "--model",
                    model_name,
                    "--root",
                    str(root),
                    "--seed",
                    str(seed),
                    "--repeat-index",
                    str(repeat_index),
                    "--result",
                    str(result_path),
                    "--effective-config",
                    str(effective_config_path),
                ]
                environment = dict(os.environ)
                environment["MPLBACKEND"] = "Agg"
                environment["PYTHONHASHSEED"] = str(seed)
                print(
                    f"[{current}/{total}] {model_name} | seed={seed} | "
                    f"repeat={repeat_index + 1}",
                    flush=True,
                )
                try:
                    with log_path.open("w", encoding="utf-8") as log_stream:
                        completed = subprocess.run(
                            command,
                            cwd=root,
                            env=environment,
                            stdout=log_stream,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout_seconds,
                            check=False,
                        )
                    result = load_json(result_path) if result_path.exists() else {
                        "status": "failed",
                        "model_name": model_name,
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "error_type": "MissingResult",
                        "error": f"Worker exited with code {completed.returncode} without a result.",
                    }
                except subprocess.TimeoutExpired:
                    result = {
                        "status": "failed",
                        "model_name": model_name,
                        "seed": seed,
                        "repeat_index": repeat_index,
                        "error_type": "TimeoutExpired",
                        "error": f"Run exceeded {args.timeout_seconds} seconds.",
                    }
                    write_json(result_path, result)
                except KeyboardInterrupt:
                    interrupted = True
                    interrupted_run = run_name
                    print(
                        "\nBenchmark interrupted. Aggregating all completed runs; "
                        "use --resume to continue this directory later.",
                        flush=True,
                    )
                    break
                if result.get("status") == "success":
                    results.append(result)
                else:
                    failures.append(result)
                    print(
                        f"  FAILED: {result.get('error_type')} - "
                        f"{result.get('error')}",
                        flush=True,
                    )
                    if args.fail_fast:
                        break
            if interrupted or (args.fail_fast and failures):
                break
        if interrupted or (args.fail_fast and failures):
            break

    raw_rows = [flatten_dict(result) for result in results]
    failure_rows = [flatten_dict(failure) for failure in failures]
    skipped_rows = [flatten_dict(item) for item in skipped]
    write_csv(output_directory / "raw_runs.csv", raw_rows)
    write_csv(output_directory / "failures.csv", failure_rows)
    write_csv(output_directory / "skipped.csv", skipped_rows)
    summary_rows = aggregate(results)
    write_csv(output_directory / "summary.csv", summary_rows)
    write_csv(output_directory / "summary_wide.csv", build_wide_summary(summary_rows))
    comparison_rows = build_model_comparison(summary_rows)
    write_csv(output_directory / "model_comparison.csv", comparison_rows)
    robustness_rows = build_robustness_comparison(summary_rows)
    write_csv(output_directory / "robustness_comparison.csv", robustness_rows)
    manifest["completed_runs"] = len(results)
    manifest["failed_runs"] = len(failures)
    manifest["skipped_runs"] = len(skipped)
    manifest["interrupted"] = interrupted
    manifest["interrupted_run"] = interrupted_run
    finished_at = datetime.now(timezone.utc).isoformat()
    if interrupted:
        manifest["status"] = "interrupted"
        manifest["interrupted_at"] = finished_at
    else:
        manifest["status"] = "completed_with_failures" if failures else "completed"
        manifest["completed_at"] = finished_at
    write_json(output_directory / "manifest.json", manifest)
    generate_markdown_report(
        output_directory / "REPORT.md",
        results,
        summary_rows,
        comparison_rows,
        robustness_rows,
        failures,
        skipped,
        manifest,
    )
    if args.plot:
        create_optional_plot(output_directory, summary_rows)

    print(
        f"Completed runs: {len(results)} | failed runs: {len(failures)} | "
        f"skipped runs: {len(skipped)}"
    )
    print(f"Results: {output_directory}")
    if interrupted:
        return 130
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
