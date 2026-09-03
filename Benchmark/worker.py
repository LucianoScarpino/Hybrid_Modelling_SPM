#!/usr/bin/env python3
"""Execute one isolated benchmark run and write a structured JSON result."""

from __future__ import annotations

import argparse
from importlib import import_module
import os
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from common import (
    PROJECT_ROOT,
    benchmark_callable,
    count_parameters,
    deep_merge,
    error_metrics,
    field_metrics,
    frame_to_field,
    load_json,
    physics_metrics,
    seed_everything,
    select_device,
    spherical_average,
    synchronize_device,
    temporal_values,
    write_json,
)


MODEL_NAMES = (
    "reference_fvm",
    "simplified_fvm",
    "fvm_hybrid",
    "pinn",
    "hybrid_pinn",
    "thermal_sindyc",
    "thermal_ffn",
)


def add_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def import_symbol(module_folder: Path, module_name: str, symbol_name: str) -> Any:
    """Load a symbol from one of the legacy, non-package model folders."""
    add_import_path(module_folder)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def simulation_arguments(config: dict[str, Any], model: str) -> Namespace:
    source = dict(config["simulation"])
    reference_ds = source.pop("reference_Ds")
    simplified_ds = source.pop("simplified_Ds")
    source["Ds_type"] = "adaptive" if model == "reference_fvm" else "constant"
    source["Ds"] = reference_ds if model == "reference_fvm" else simplified_ds
    source["variant_number"] = None
    return Namespace(**source)


def stored_reference(root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    path = root / "Single_Particle_Model_V3" / "SPMDataset" / "full_simulation_dataset.csv"
    frame = pd.read_csv(path)
    field, radius, tau = frame_to_field(frame, "concentration")
    return frame, field, radius, tau


def solution_accuracy(
    solution: dict[str, Any],
    reference_field: np.ndarray,
    reference_radius: np.ndarray,
    reference_time: np.ndarray,
) -> dict[str, Any]:
    radius = np.asarray(solution["radius"], dtype=float)
    physical_time = np.asarray(solution["time"], dtype=float)
    if (
        not np.allclose(radius, reference_radius)
        or not np.allclose(physical_time, reference_time)
    ):
        raise ValueError(
            "Simulation and stored-reference physical grids are not aligned."
        )
    return field_metrics(
        reference_field,
        np.asarray(solution["concentration"], dtype=float),
        radius,
    )


def run_fvm(
    model_name: str,
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    module_folder = root / "Single_Particle_Model_V3"
    PostProcessing = import_symbol(module_folder, "DatasetGenerator", "PostProcessing")
    Simulate = import_symbol(module_folder, "ParticleSimulation", "Simulate")

    arguments = simulation_arguments(config, model_name)
    timing_config = config["timing"]

    def simulate():
        return Simulate(arguments).run()

    solution, solver_timing = benchmark_callable(
        simulate,
        repetitions=int(timing_config["solver_repetitions"]),
        warmup=int(timing_config["warmup"]),
    )
    post = PostProcessing(**solution)
    dataset_assembly_start = perf_counter()
    dataset, _ = post.generate_dataset()
    dataset_assembly_seconds = perf_counter() - dataset_assembly_start
    residual = post.compute_mass_balance_residual()
    reference_frame, reference_field, reference_radius, _ = stored_reference(root)
    reference_time = temporal_values(reference_frame, "time")

    return {
        "device": "cpu",
        "timing": {
            "training_seconds": 0.0,
            "dataset_assembly_seconds": dataset_assembly_seconds,
            "offline_dataset_generation_seconds": (
                solver_timing["mean_seconds"] + dataset_assembly_seconds
            ),
            "online_inference": solver_timing,
            "online_end_to_end": solver_timing,
        },
        "accuracy": solution_accuracy(
            solution,
            reference_field,
            reference_radius,
            reference_time,
        ),
        "physics": {
            **physics_metrics(
                solution["concentration"],
                solution["radius"],
                solution["tau"],
                solution["dimensionless_fluxes"],
                solution["initial_concentration"],
            ),
            "postprocessor_mass_balance_max_abs": float(np.max(np.abs(residual))),
            "temperature_minimum_kelvin": float(np.min(solution["temperatures"])),
            "temperature_maximum_kelvin": float(np.max(solution["temperatures"])),
        },
        "model": {
            "trainable_parameters": 0,
            "radial_nodes": int(len(solution["radius"])),
            "time_points": int(len(solution["tau"])),
            "field_points": int(np.asarray(solution["concentration"]).size),
            "dataset_rows": int(len(dataset["rho"])),
            "diffusivity_type": arguments.Ds_type,
            "diffusivity": float(arguments.Ds),
        },
    }


def full_profile_features(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = (
        dataset.pivot(index="tau", columns="rho", values="concentration")
        .sort_index()
        .sort_index(axis=1)
    )
    states = (
        dataset.drop_duplicates("tau")
        .set_index("tau")
        .sort_index()
        .loc[profiles.index]
    )
    features = states[
        ["average_concentration", "surface_concentration", "current"]
    ].to_numpy(dtype=np.float32)
    return (
        features,
        profiles.to_numpy(dtype=float),
        profiles.columns.to_numpy(dtype=float),
    )


def run_fvm_hybrid(
    root: Path,
    config: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    import torch

    module_folder = root / "Hybrid_Model"
    Loader = import_symbol(module_folder, "DatasetLoader", "Loader")
    Processing = import_symbol(module_folder, "Processor", "Processing")

    model_config = config["fvm_hybrid"]
    timing_config = config["timing"]
    dataset_folder = module_folder / "Dataset"
    # This very small network is normally faster on CPU because dispatching
    # its tiny batches to MPS costs more than the matrix operations themselves.
    device = select_device(model_config.get("device", config["device"]))

    loader = Loader(random_state=int(config["split_seed"]))
    simplified_dataset = loader.get_simplified_dataset(dataset_folder)
    reference_dataset = loader.get_reference_dataset(dataset_folder)
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = (
        loader.split_dataset(reference_dataset, simplified_dataset)
    )
    batch_size = int(model_config["batch_size"])
    train_loader = loader.loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    validation_loader = loader.loader(
        x_validation, y_validation, batch_size=batch_size
    )

    inventory_offset = Processing.estimate_inventory_offset(y_train, loader.radius)
    processor = Processing(
        train_loader,
        validation_loader,
        loader.radius,
        inventory_offset,
    )
    synchronize_device(device)
    training_start = perf_counter()
    old_working_directory = Path.cwd()
    os.chdir(run_directory)
    try:
        model, training_metrics = processor.train(
            lr=float(model_config["learning_rate"]),
            epochs=int(model_config["epochs"]),
            early_stopping_patience=int(model_config["early_stopping_patience"]),
            input_dim=3,
            hidden_dim=int(model_config["hidden_dim"]),
            out_dim=len(loader.radius),
            device=device,
            save_model_checkpoint=False,
        )
    finally:
        os.chdir(old_working_directory)
    synchronize_device(device)
    training_seconds = perf_counter() - training_start
    model.eval()

    full_features, simplified_profiles, radius = full_profile_features(simplified_dataset)
    feature_tensor = torch.tensor(full_features, dtype=torch.float32, device=device)

    def predict_full_profiles():
        with torch.inference_mode():
            raw_correction = model(feature_tensor)
            correction, _, _ = processor.apply_inventory_constraint(raw_correction)
        return correction.detach().cpu().numpy()

    corrections, correction_timing = benchmark_callable(
        predict_full_profiles,
        repetitions=int(timing_config["inference_repetitions"]),
        warmup=int(timing_config["warmup"]),
        device=device,
    )
    hybrid_time_by_radius = simplified_profiles + corrections
    hybrid_field = hybrid_time_by_radius.T
    reference_field, reference_radius, reference_tau = frame_to_field(
        reference_dataset, "concentration"
    )
    if not np.allclose(radius, reference_radius):
        raise ValueError("Hybrid output and reference radial grids are not aligned.")
    reference_flux = temporal_values(reference_dataset, "dimensionless_flux")
    initial_concentration = float(config["simulation"]["C0"])

    test_features = x_test.to(device)
    with torch.inference_mode():
        raw_test_correction = model(test_features)
        test_correction, _, _ = processor.apply_inventory_constraint(raw_test_correction)
    test_correction = test_correction.detach().cpu().numpy()
    test_baseline = loader.get_simplified_profiles("test").numpy()
    test_reference = test_baseline + y_test.numpy()
    test_hybrid = test_baseline + test_correction

    # Include the physical baseline in an end-to-end online measurement.
    fvm_folder = root / "Single_Particle_Model_V3"
    Simulate = import_symbol(fvm_folder, "ParticleSimulation", "Simulate")

    simulation_args = simulation_arguments(config, "simplified_fvm")

    def predict_end_to_end():
        solution = Simulate(simulation_args).run()
        field = np.asarray(solution["concentration"], dtype=float)
        local_radius = np.asarray(solution["radius"], dtype=float)
        average = spherical_average(field, local_radius)
        features = np.column_stack(
            (average, field[-1, :], np.asarray(solution["currents"], dtype=float))
        ).astype(np.float32)
        with torch.inference_mode():
            features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            raw = model(features_tensor)
            constrained, _, _ = processor.apply_inventory_constraint(raw)
        corrected = field.T + constrained.detach().cpu().numpy()
        return corrected.T, solution

    end_to_end_result, end_to_end_timing = benchmark_callable(
        predict_end_to_end,
        repetitions=int(timing_config["end_to_end_repetitions"]),
        warmup=int(timing_config["warmup"]),
        device=device,
    )
    end_to_end_field, end_to_end_solution = end_to_end_result

    robustness: dict[str, Any] = {}
    if bool(model_config.get("evaluate_variants", True)):
        variants_folder = dataset_folder / "Variants"
        for variant_path in sorted(
            variants_folder.glob("simplified_simulation_dataset_*.csv")
        ):
            variant_number = variant_path.stem.rsplit("_", 1)[-1]
            variant_dataset = pd.read_csv(variant_path)
            variant_features, variant_baseline, variant_radius = full_profile_features(
                variant_dataset
            )
            if not np.allclose(variant_radius, reference_radius):
                raise ValueError(
                    f"Variant {variant_number} has a different radial grid."
                )
            with torch.inference_mode():
                variant_tensor = torch.tensor(
                    variant_features, dtype=torch.float32, device=device
                )
                raw_variant_correction = model(variant_tensor)
                variant_correction, _, _ = processor.apply_inventory_constraint(
                    raw_variant_correction
                )
            variant_hybrid = (
                variant_baseline + variant_correction.detach().cpu().numpy()
            ).T
            variant_baseline_field = variant_baseline.T
            baseline_metrics = field_metrics(
                reference_field, variant_baseline_field, reference_radius
            )
            hybrid_metrics = field_metrics(
                reference_field, variant_hybrid, reference_radius
            )
            baseline_rmse = baseline_metrics["field"]["rmse"]
            hybrid_rmse = hybrid_metrics["field"]["rmse"]
            robustness[f"variant_{variant_number}"] = {
                "source": str(variant_path.relative_to(root)),
                "initial_simplified_concentration": float(
                    variant_dataset["concentration"].iloc[0]
                ),
                "dimensionless_final_time": float(variant_dataset["tau"].max()),
                "baseline_accuracy": baseline_metrics,
                "hybrid_accuracy": hybrid_metrics,
                "field_rmse_reduction_percent": (
                    100.0 * (1.0 - hybrid_rmse / baseline_rmse)
                    if baseline_rmse > 0.0
                    else None
                ),
                "physics_against_reference_inventory": physics_metrics(
                    variant_hybrid,
                    reference_radius,
                    reference_tau,
                    reference_flux,
                    initial_concentration,
                ),
            }

    return {
        "device": str(device),
        "timing": {
            "training_seconds": training_seconds,
            "processor_training_seconds": float(processor.training_time_seconds),
            "online_correction": correction_timing,
            "online_end_to_end": end_to_end_timing,
            "online_correction_per_profile_seconds": (
                correction_timing["mean_seconds"] / len(full_features)
            ),
        },
        "accuracy": field_metrics(reference_field, hybrid_field, reference_radius),
        "end_to_end_accuracy": field_metrics(
            reference_field, end_to_end_field, reference_radius
        ),
        "test_accuracy": {
            "field": error_metrics(test_reference, test_hybrid),
            "baseline_field": error_metrics(test_reference, test_baseline),
        },
        "physics": physics_metrics(
            hybrid_field,
            reference_radius,
            reference_tau,
            reference_flux,
            initial_concentration,
        ),
        "end_to_end_physics": physics_metrics(
            end_to_end_field,
            end_to_end_solution["radius"],
            end_to_end_solution["tau"],
            end_to_end_solution["dimensionless_fluxes"],
            initial_concentration,
        ),
        "training": {
            **training_metrics,
            "inventory_offset": float(inventory_offset),
            "epochs_requested": int(model_config["epochs"]),
            "epochs_completed": int(processor.completed_training_epochs),
            "best_epoch": int(training_metrics["best_epoch"]),
            "early_stopped": bool(processor.early_stopped),
        },
        "robustness": robustness,
        "model": {
            "trainable_parameters": count_parameters(model),
            "radial_nodes": int(len(radius)),
            "time_points": int(len(reference_tau)),
            "test_profiles": int(len(x_test)),
        },
    }


def split_indices(length: int, split_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(split_seed)
    shuffled = generator.permutation(length)
    temporary_count = int(np.ceil(0.30 * length))
    test_count = int(np.ceil(0.50 * temporary_count))
    train = shuffled[:-temporary_count]
    temporary = shuffled[-temporary_count:]
    validation = temporary[:-test_count]
    test = temporary[-test_count:]
    return train, validation, test


def tensor_loader(
    inputs: np.ndarray,
    targets: np.ndarray | None,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    input_tensor = torch.tensor(inputs[indices], dtype=torch.float32)
    tensors = [input_tensor]
    if targets is not None:
        tensors.append(torch.tensor(targets[indices], dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def pinn_loaders(
    root: Path,
    model_seed: int,
    split_seed: int,
    batch_size: int,
    tau_final: float,
):
    from scipy.stats import qmc

    data_path = root / "PINN" / "Data" / "pinn_training_dataset.csv"
    data_frame = pd.read_csv(data_path)
    data = data_frame[["rho", "tau"]].to_numpy(dtype=np.float32)
    labels = data_frame[["concentration"]].to_numpy(dtype=np.float32)
    collocation_frame = data_frame.loc[
        (data_frame["rho"] > 0.0)
        & (data_frame["rho"] < 1.0)
        & (data_frame["tau"] > 0.0),
        ["rho", "tau"],
    ]
    collocation = collocation_frame.to_numpy(dtype=np.float32)
    sobol = qmc.Sobol(d=1, scramble=True, seed=split_seed)
    sobol_power = int(np.ceil(np.log2(len(data))))
    boundary_tau = (
        tau_final
        * sobol.random_base2(sobol_power)[: len(data)].astype(np.float32)
    )
    boundary = np.column_stack(
        (np.ones(len(data), dtype=np.float32), boundary_tau[:, 0])
    )

    data_split = split_indices(len(data), split_seed)
    collocation_split = split_indices(len(collocation), split_seed)
    boundary_split = split_indices(len(boundary), split_seed)
    data_loaders = tuple(
        tensor_loader(
            data,
            labels,
            indices,
            batch_size,
            split_name == 0,
            model_seed + split_name,
        )
        for split_name, indices in enumerate(data_split)
    )
    collocation_loaders = tuple(
        tensor_loader(
            collocation,
            None,
            indices,
            batch_size,
            split_name == 0,
            model_seed + 10 + split_name,
        )
        for split_name, indices in enumerate(collocation_split)
    )
    boundary_loaders = tuple(
        tensor_loader(
            boundary,
            None,
            indices,
            batch_size,
            split_name == 0,
            model_seed + 20 + split_name,
        )
        for split_name, indices in enumerate(boundary_split)
    )
    return data_frame, data_loaders, collocation_loaders, boundary_loaders


def predict_pinn_grid(
    model,
    coordinates: np.ndarray,
    initial_concentration: float,
    device,
    batch_size: int = 4096,
) -> np.ndarray:
    import torch

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(coordinates), batch_size):
            batch = torch.tensor(
                coordinates[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(model(batch, initial_concentration).detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).ravel()


def prediction_field(
    reference_frame: pd.DataFrame,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = reference_frame[["rho", "tau"]].copy()
    frame["prediction"] = prediction
    return frame_to_field(frame, "prediction")


def run_pinn(
    root: Path,
    config: dict[str, Any],
    seed: int,
    run_directory: Path,
) -> dict[str, Any]:
    import torch

    module_folder = root / "PINN"
    Processing = import_symbol(module_folder, "Processor", "Processing")
    Testing = import_symbol(module_folder, "Tester", "Testing")

    model_config = config["pinn"]
    timing_config = config["timing"]
    device = select_device(model_config.get("device", config["device"]))
    batch_size = int(model_config["batch_size"])
    _, data_loaders, collocation_loaders, boundary_loaders = pinn_loaders(
        root,
        seed,
        int(config["split_seed"]),
        batch_size,
        float(model_config["tau_final"]),
    )
    processor = Processing(
        data_loaders[0],
        collocation_loaders[0],
        boundary_loaders[0],
        data_loaders[1],
        collocation_loaders[1],
        boundary_loaders[1],
        device,
    )
    synchronize_device(device)
    training_start = perf_counter()
    old_working_directory = Path.cwd()
    os.chdir(run_directory)
    try:
        model = processor.train(
            epochs=int(model_config["epochs"]),
            init_concentration=float(model_config["initial_concentration"]),
            lambda_d=float(model_config["lambda_data"]),
            lambda_f=float(model_config["lambda_pde"]),
            lambda_b=float(model_config["lambda_boundary"]),
            flux=float(model_config["surface_flux"]),
            lr=float(model_config["learning_rate"]),
            early_stopping_patience=int(model_config["early_stopping_patience"]),
            sheduler_patience=int(model_config["scheduler_patience"]),
            lbfgs_max_iter=int(model_config["lbfgs_max_iterations"]),
            validation_interval=int(model_config.get("validation_interval", 1)),
            diagnostics_interval=int(model_config.get("diagnostics_interval", 1)),
            early_stopping_min_delta=float(
                model_config.get("early_stopping_min_delta", 0.0)
            ),
        )
    finally:
        os.chdir(old_working_directory)
    synchronize_device(device)
    training_seconds = perf_counter() - training_start

    held_out_metrics = Testing(
        data_loaders[2], collocation_loaders[2], boundary_loaders[2]
    ).test(
        model,
        float(model_config["initial_concentration"]),
        float(model_config["surface_flux"]),
    )

    reference_frame, reference_field, radius, tau = stored_reference(root)
    coordinates = reference_frame[["rho", "tau"]].to_numpy(dtype=np.float32)

    def predict_grid():
        return predict_pinn_grid(
            model,
            coordinates,
            float(model_config["initial_concentration"]),
            device,
        )

    prediction, inference_timing = benchmark_callable(
        predict_grid,
        repetitions=int(timing_config["inference_repetitions"]),
        warmup=int(timing_config["warmup"]),
        device=device,
    )
    predicted_field, predicted_radius, predicted_tau = prediction_field(
        reference_frame, prediction
    )
    if not np.allclose(radius, predicted_radius) or not np.allclose(tau, predicted_tau):
        raise ValueError("PINN prediction and reference grids are not aligned.")
    reference_flux = temporal_values(reference_frame, "dimensionless_flux")
    constant_flux = np.full_like(tau, float(model_config["surface_flux"]))

    return {
        "device": str(device),
        "timing": {
            "training_seconds": training_seconds,
            "online_inference": inference_timing,
            "online_inference_per_coordinate_seconds": (
                inference_timing["mean_seconds"] / len(coordinates)
            ),
        },
        "accuracy": field_metrics(reference_field, predicted_field, radius),
        "held_out_accuracy": held_out_metrics,
        "physics": physics_metrics(
            predicted_field,
            radius,
            tau,
            reference_flux,
            float(model_config["initial_concentration"]),
        ),
        "model_flux_physics": physics_metrics(
            predicted_field,
            radius,
            tau,
            constant_flux,
            float(model_config["initial_concentration"]),
        ),
        "training": {
            "epochs_requested": int(model_config["epochs"]),
            "split_seed": int(config["split_seed"]),
            **getattr(processor, "training_summary", {}),
        },
        "model": {
            "trainable_parameters": count_parameters(model),
            "field_points": int(len(coordinates)),
            "radial_nodes": int(len(radius)),
            "time_points": int(len(tau)),
        },
    }


def find_pinn_checkpoint(root: Path, configured: str | None) -> Path:
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = list((root / "PINN" / "Results" / "Models").glob("pinn_checkpoint*.pth"))
    if not candidates:
        raise FileNotFoundError("No PINN checkpoint was found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def residual_dataset(model, loader, initial_concentration: float, device):
    import torch
    from torch.utils.data import TensorDataset

    inputs = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for coordinates, reference in loader:
            coordinates_device = coordinates.to(device)
            reference_device = reference.to(device)
            baseline = model(coordinates_device, initial_concentration)
            inputs.append(torch.cat((coordinates_device, baseline), dim=1).cpu())
            targets.append((reference_device - baseline).cpu())
    return TensorDataset(torch.cat(inputs), torch.cat(targets))


def residual_loader(dataset, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def run_hybrid_pinn(
    root: Path,
    config: dict[str, Any],
    seed: int,
    run_directory: Path,
) -> dict[str, Any]:
    import torch

    module_folder = root / "PINN"
    HybridModel = import_symbol(module_folder, "NeuralNetwork", "HybridModel")
    MLP = import_symbol(module_folder, "NeuralNetwork", "MLP")
    ResidualLearner = import_symbol(
        module_folder, "ResidualLearning", "ResidualLearner"
    )

    model_config = config["hybrid_pinn"]
    pinn_config = config["pinn"]
    timing_config = config["timing"]
    device = select_device(model_config.get("device", config["device"]))
    checkpoint_path = find_pinn_checkpoint(root, model_config.get("checkpoint"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    initial_concentration = float(
        checkpoint.get("initial_concentration", pinn_config["initial_concentration"])
    )
    pinn_model = MLP().to(device)
    pinn_model.load_state_dict(checkpoint["model_state_dict"])
    pinn_model.eval().requires_grad_(False)

    batch_size = int(model_config["batch_size"])
    _, data_loaders, _, _ = pinn_loaders(
        root,
        seed,
        int(config["split_seed"]),
        batch_size,
        float(pinn_config["tau_final"]),
    )
    synchronize_device(device)
    residual_data_start = perf_counter()
    residual_datasets = tuple(
        residual_dataset(pinn_model, loader, initial_concentration, device)
        for loader in data_loaders
    )
    synchronize_device(device)
    residual_data_seconds = perf_counter() - residual_data_start
    residual_loaders = (
        residual_loader(residual_datasets[0], batch_size, True, seed + 30),
        residual_loader(residual_datasets[1], batch_size, False, seed + 31),
        residual_loader(residual_datasets[2], batch_size, False, seed + 32),
    )

    learner = ResidualLearner()
    synchronize_device(device)
    training_start = perf_counter()
    old_working_directory = Path.cwd()
    os.chdir(run_directory)
    try:
        residual_model = learner.learn_residual(
            residual_loaders[0],
            residual_loaders[1],
            epochs=int(model_config["epochs"]),
            lr=float(model_config["learning_rate"]),
            early_stopping_patience=int(model_config["early_stopping_patience"]),
            device=device,
            validation_interval=int(model_config.get("validation_interval", 1)),
            early_stopping_min_delta=float(
                model_config.get("early_stopping_min_delta", 0.0)
            ),
        )
    finally:
        os.chdir(old_working_directory)
    synchronize_device(device)
    training_seconds = perf_counter() - training_start
    hybrid_model = HybridModel(pinn_model, residual_model).to(device)
    hybrid_model.eval().requires_grad_(False)

    reference_frame, reference_field, radius, tau = stored_reference(root)
    coordinates = reference_frame[["rho", "tau"]].to_numpy(dtype=np.float32)

    def predict_grid():
        return predict_pinn_grid(
            hybrid_model,
            coordinates,
            initial_concentration,
            device,
        )

    prediction, inference_timing = benchmark_callable(
        predict_grid,
        repetitions=int(timing_config["inference_repetitions"]),
        warmup=int(timing_config["warmup"]),
        device=device,
    )
    predicted_field, _, _ = prediction_field(reference_frame, prediction)
    reference_flux = temporal_values(reference_frame, "dimensionless_flux")

    test_inputs, test_targets = residual_datasets[2].tensors
    with torch.inference_mode():
        test_prediction = residual_model(test_inputs.to(device)).cpu().numpy()
    test_reference = (test_inputs[:, 2:3] + test_targets).numpy()
    test_hybrid = test_inputs[:, 2:3].numpy() + test_prediction

    diagnostics: dict[str, Any] = {}
    if bool(model_config.get("physics_diagnostics", False)):
        corrected_frame = reference_frame.rename(
            columns={"concentration": "concentration_reference"}
        ).copy()
        corrected_frame["concentration_hybrid"] = prediction
        diagnostics["hard_constraints"] = learner.compute_hard_constraint_checks(
            hybrid_model, corrected_frame, initial_concentration, device
        )
        diagnostics["surface_flux"] = learner.compute_surface_flux_check(
            hybrid_model, corrected_frame, initial_concentration, device
        )
        pde = learner.compute_pde_consistency_checks(
            hybrid_model, corrected_frame, initial_concentration, device
        )
        diagnostics["pde"] = {
            "number_of_points": pde["number_of_points"],
            "full_physics": pde["full_physics"],
            "simplified_physics": pde["simplified_physics"],
            "regularity": pde["regularity"],
        }

    return {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "timing": {
            "residual_dataset_seconds": residual_data_seconds,
            "training_seconds": training_seconds,
            "correction_pipeline_seconds": residual_data_seconds + training_seconds,
            "online_inference": inference_timing,
            "online_inference_per_coordinate_seconds": (
                inference_timing["mean_seconds"] / len(coordinates)
            ),
        },
        "accuracy": field_metrics(reference_field, predicted_field, radius),
        "test_accuracy": error_metrics(test_reference, test_hybrid),
        "physics": physics_metrics(
            predicted_field,
            radius,
            tau,
            reference_flux,
            initial_concentration,
        ),
        "diagnostics": diagnostics,
        "training": {
            "epochs_requested": int(model_config["epochs"]),
            "split_seed": int(config["split_seed"]),
            **getattr(learner, "training_summary", {}),
        },
        "model": {
            "frozen_pinn_parameters": count_parameters(pinn_model),
            "trainable_parameters": count_parameters(residual_model),
            "total_parameters": count_parameters(pinn_model) + count_parameters(residual_model),
            "field_points": int(len(coordinates)),
        },
    }


def run_thermal(
    model_name: str,
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch

    module_folder = root / "Hybrid_Model"
    FFNDiscovering = import_symbol(module_folder, "ThermalDiscover", "FFNDiscovering")
    SINDyc = import_symbol(module_folder, "ThermalDiscover", "SINDyc")

    thermal_config = config["thermal"]
    timing_config = config["timing"]
    data_folder = module_folder / "Dataset" / "ForThermalVariants"
    names = [f"full_simulation_dataset_{index}.csv" for index in range(8)]
    raw_datasets = [pd.read_csv(data_folder / name) for name in names]
    ambient = float(thermal_config["ambient_temperature"])
    rollout_repetitions = int(timing_config["inference_repetitions"])
    rollout_warmup = int(timing_config["warmup"])

    if model_name == "thermal_sindyc":
        discoverer = SINDyc(raw_datasets[:5])
        training_start = perf_counter()
        model, _, selected_threshold = discoverer.discover_equation(
            thermal_config["thresholds"],
            validation_dataset=raw_datasets[5],
            T_amb=ambient,
        )
        training_seconds = perf_counter() - training_start
        results: dict[str, Any] = {}
        rollout_timings: dict[str, Any] = {}
        for test_type, dataset_index in (("interpolation", 6), ("extrapolation", 7)):
            evaluated = discoverer.evaluate_dataset(
                model, raw_datasets[dataset_index], T_amb=ambient
            )

            def rollout():
                signals = discoverer._signals(raw_datasets[dataset_index], ambient)
                return discoverer._predict_temperatures(
                    model, signals[4], ambient, signals[1], signals[3]
                )

            _, rollout_timing = benchmark_callable(
                rollout,
                repetitions=rollout_repetitions,
                warmup=rollout_warmup,
            )
            results[test_type] = evaluated["metrics"]
            rollout_timings[test_type] = rollout_timing
        return {
            "device": "cpu",
            "timing": {
                "training_seconds": training_seconds,
                "rollout": rollout_timings,
            },
            "thermal_accuracy": results,
            "training": {"selected_threshold": float(selected_threshold)},
            "model": {"active_terms": int(model.complexity), "trainable_parameters": 0},
        }

    device = select_device(thermal_config.get("device", config["device"]))
    discoverer = FFNDiscovering(data_folder, names, T_amb=ambient)
    train_loader, validation_loader, interpolation_loader, extrapolation_loader = (
        discoverer.generate_thermal_loaders()
    )
    synchronize_device(device)
    training_start = perf_counter()
    model = discoverer.predict_equations(
        device=device,
        lr=float(thermal_config["ffn_learning_rate"]),
        training_epochs=int(thermal_config["ffn_epochs"]),
        early_stopping_epochs=int(thermal_config["ffn_early_stopping_patience"]),
        train_loader=train_loader,
        val_loader=validation_loader,
    )
    synchronize_device(device)
    training_seconds = perf_counter() - training_start
    results = {}
    rollout_timings = {}
    for test_type, test_loader in (
        ("interpolation", interpolation_loader),
        ("extrapolation", extrapolation_loader),
    ):
        evaluated = discoverer.test_thermalFNN(
            model, test_loader, device, test_type=test_type, T_amb=ambient
        )

        def rollout():
            return discoverer.temperature_rollout(
                model,
                evaluated["time"],
                evaluated["current"],
                evaluated["reference_temperature"][0],
                device,
                ambient,
            )

        _, rollout_timing = benchmark_callable(
            rollout,
            repetitions=rollout_repetitions,
            warmup=rollout_warmup,
            device=device,
        )
        results[test_type] = evaluated["metrics"]
        rollout_timings[test_type] = rollout_timing
    return {
        "device": str(device),
        "timing": {
            "training_seconds": training_seconds,
            "rollout": rollout_timings,
        },
        "thermal_accuracy": results,
        "training": {"epochs_requested": int(thermal_config["ffn_epochs"])},
        "model": {"trainable_parameters": count_parameters(model)},
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    default_config = load_json(Path(__file__).with_name("default_config.json"))
    config = default_config
    if args.config:
        config = deep_merge(config, load_json(Path(args.config)))
    if args.effective_config:
        config = deep_merge(config, load_json(Path(args.effective_config)))

    seed_everything(args.seed, bool(config.get("deterministic_algorithms", False)))
    run_directory = Path(args.result).resolve().parent
    run_directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    if args.model in {"reference_fvm", "simplified_fvm"}:
        payload = run_fvm(args.model, root, config)
    elif args.model == "fvm_hybrid":
        payload = run_fvm_hybrid(root, config, run_directory)
    elif args.model == "pinn":
        payload = run_pinn(root, config, args.seed, run_directory)
    elif args.model == "hybrid_pinn":
        payload = run_hybrid_pinn(root, config, args.seed, run_directory)
    elif args.model in {"thermal_sindyc", "thermal_ffn"}:
        payload = run_thermal(args.model, root, config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return {
        "status": "success",
        "model_name": args.model,
        "seed": int(args.seed),
        "repeat_index": int(args.repeat_index),
        "worker_wall_seconds": perf_counter() - start,
        **payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--effective-config", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = execute(args)
        write_json(args.result, result)
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "model_name": args.model,
            "seed": int(args.seed),
            "repeat_index": int(args.repeat_index),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.result, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
