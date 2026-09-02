"""Shared numerical, timing, and serialization helpers for benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(to_jsonable(data), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str = "auto"):
    import torch

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps.")
    return torch.device(requested)


def synchronize_device(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timing_statistics(samples: list[float]) -> dict[str, float | int | list[float]]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("At least one timing sample is required.")
    return {
        "n": int(values.size),
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median_seconds": float(np.median(values)),
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "p05_seconds": float(np.percentile(values, 5)),
        "p95_seconds": float(np.percentile(values, 95)),
        "samples_seconds": values.tolist(),
    }


def benchmark_callable(
    function: Callable[[], Any],
    repetitions: int,
    warmup: int = 1,
    device=None,
) -> tuple[Any, dict[str, Any]]:
    """Warm up and time a callable, synchronizing accelerators when needed."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")

    result = None
    for _ in range(warmup):
        result = function()
    if device is not None:
        synchronize_device(device)

    samples: list[float] = []
    for _ in range(repetitions):
        if device is not None:
            synchronize_device(device)
        start = perf_counter()
        result = function()
        if device is not None:
            synchronize_device(device)
        samples.append(perf_counter() - start)
    return result, timing_statistics(samples)


def error_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Reference shape {reference.shape} does not match prediction "
            f"shape {prediction.shape}."
        )
    error = prediction - reference
    norm = np.linalg.norm(reference)
    return {
        "relative_l2": float(np.linalg.norm(error) / norm) if norm else float("nan"),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def spherical_average(field: np.ndarray, radius: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    radius = np.asarray(radius, dtype=float)
    numerator = np.trapezoid(field * radius[:, None] ** 2, radius, axis=0)
    denominator = np.trapezoid(radius**2, radius)
    return numerator / denominator


def field_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    radius: np.ndarray,
) -> dict[str, dict[str, float]]:
    reference_average = spherical_average(reference, radius)
    prediction_average = spherical_average(prediction, radius)
    return {
        "field": error_metrics(reference, prediction),
        "centre": error_metrics(reference[0, :], prediction[0, :]),
        "surface": error_metrics(reference[-1, :], prediction[-1, :]),
        "average": error_metrics(reference_average, prediction_average),
    }


def physics_metrics(
    field: np.ndarray,
    radius: np.ndarray,
    tau: np.ndarray,
    dimensionless_flux: np.ndarray,
    initial_concentration: float,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    from scipy.integrate import cumulative_trapezoid

    field = np.asarray(field, dtype=float)
    average = spherical_average(field, radius)
    expected = initial_concentration - 3.0 * cumulative_trapezoid(
        np.asarray(dimensionless_flux, dtype=float),
        np.asarray(tau, dtype=float),
        initial=0.0,
    )
    residual = average - expected
    violations = int(np.count_nonzero((field < 0.0) | (field > 1.0)))
    return {
        "mass_balance_max_abs": float(np.max(np.abs(residual))),
        "mass_balance_rmse": float(np.sqrt(np.mean(residual**2))),
        "mass_balance_final_residual": float(residual[-1]),
        "mass_balance_passed": bool(np.max(np.abs(residual)) < tolerance),
        "minimum_concentration": float(np.min(field)),
        "maximum_concentration": float(np.max(field)),
        "physical_bounds_violations": violations,
        "physical_bounds_violation_fraction": float(violations / field.size),
        "finite": bool(np.all(np.isfinite(field))),
    }


def frame_to_field(
    frame: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        frame.pivot(index="rho", columns="tau", values=value_column)
        .sort_index()
        .sort_index(axis=1)
    )
    return (
        pivot.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.columns.to_numpy(dtype=float),
    )


def temporal_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return (
        frame[["tau", column]]
        .drop_duplicates("tau")
        .sort_values("tau")[column]
        .to_numpy(dtype=float)
    )


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        elif isinstance(value, (list, tuple)):
            continue
        else:
            flattened[full_key] = value
    return flattened


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "status_porcelain": run("status", "--short"),
    }


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": torch.backends.mps.is_available(),
            }
        )
    except ImportError:
        metadata["torch"] = None
    metadata["numpy"] = np.__version__
    metadata["pandas"] = pd.__version__
    return metadata
