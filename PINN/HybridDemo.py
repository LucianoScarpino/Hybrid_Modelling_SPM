import torch
import pandas as pd
import numpy as np

from pathlib import Path
from time import perf_counter

from Generator import DatasetGenerator
from NeuralNetwork import HybridModel, MLP
from ResidualLearning import ResidualLearner
from Tester import Testing


#device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#paths
checkpoint_path = Path("./Results/Models/pinn_checkpoint_20260731_120647_711512.pth")
reference_dataset_path = Path("./Data/pinn_training_dataset.csv")
full_reference_dataset_path = Path("./Data/full_simulation_dataset.csv")
corrected_dataset_path = Path("./Data/corrected_dataset.csv")
checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)

#restore PINN trained model
pinn_model = MLP().to(device)
pinn_model.load_state_dict(checkpoint["model_state_dict"])
pinn_model.eval()
pinn_model.requires_grad_(False)

#create residual dataset
generator = DatasetGenerator(reference_dataset_path)
reference_frame = pd.read_csv(reference_dataset_path)
data_points = reference_frame.values
d_train_loader, d_val_loader, d_test_loader = generator.generate_dataloaders(data_points)

initial_concentration = checkpoint["initial_concentration"]

ResidualLearner.synchronize_device(device)
pinn_inference_start = perf_counter()
res_train_dataset = generator.generate_residual_dataset(pinn_model, d_train_loader, initial_concentration)
res_val_dataset = generator.generate_residual_dataset(pinn_model, d_val_loader, initial_concentration)
res_test_dataset = generator.generate_residual_dataset(pinn_model, d_test_loader, initial_concentration)
ResidualLearner.synchronize_device(device)
pinn_inference_seconds = perf_counter() - pinn_inference_start

res_train_loader,res_val_loader,res_test_loader = generator.generate_residual_loaders(
    res_train_dataset,
    res_val_dataset,
    res_test_dataset,
    batch_size=512
)

residual_epochs = 3000
residual_patience = 300
residual_lr = 1e-3

residual_learner = ResidualLearner()
ResidualLearner.synchronize_device(device)
residual_training_start = perf_counter()
residual_model = residual_learner.learn_residual(
    train_res_dataset=res_train_loader,
    val_res_dataset=res_val_loader,
    epochs=residual_epochs,
    lr=residual_lr,
    early_stopping_patience=residual_patience,
    device=device
)
ResidualLearner.synchronize_device(device)
residual_training_seconds = perf_counter() - residual_training_start

residual_model = residual_model.to(device)

tester = Testing()
hybrid_metrics, residual_predictions, residual_targets = tester.test_residuals(
    residual_model,
    res_test_loader
    )

# Reassemble the complete grid using the PINN predictions already produced
# while building the train, validation and test residual datasets.
all_residual_inputs = torch.cat(
    (
        res_train_dataset.tensors[0],
        res_val_dataset.tensors[0],
        res_test_dataset.tensors[0]
    ),
    dim=0
)
all_inputs_numpy = all_residual_inputs.numpy()
grid_order = np.lexsort(
    (
        all_inputs_numpy[:, 0],
        all_inputs_numpy[:, 1]
    )
)
ordered_residual_inputs = all_residual_inputs[grid_order]

residual_model.eval()
residual_corrections = []
hybrid_concentrations = []
correction_batch_size = 512

ResidualLearner.synchronize_device(device)
correction_start = perf_counter()

with torch.inference_mode():
    for start_index in range(0,len(ordered_residual_inputs),correction_batch_size):
        end_index = start_index + correction_batch_size
        residual_inputs_batch = ordered_residual_inputs[start_index:end_index].to(device)
        residual_correction_batch = residual_model(residual_inputs_batch)
        concentration_hybrid_batch = (residual_inputs_batch[:, 2:3] + residual_correction_batch)

        residual_corrections.append(residual_correction_batch.cpu())
        hybrid_concentrations.append(concentration_hybrid_batch.cpu())

ResidualLearner.synchronize_device(device)
correction_seconds = perf_counter() - correction_start

residual_corrections = torch.cat(residual_corrections,dim=0).numpy().ravel()
hybrid_concentrations = torch.cat(hybrid_concentrations,dim=0).numpy().ravel()
ordered_inputs_numpy = ordered_residual_inputs.numpy()

full_reference_frame = pd.read_csv(full_reference_dataset_path)
full_reference_frame = (full_reference_frame.sort_values(["tau", "rho"]).reset_index(drop=True))

if len(full_reference_frame) != len(ordered_inputs_numpy):
    raise ValueError(
        "The full FVM dataset and the corrected prediction grid "
        "have different numbers of points."
    )

if not (
    np.allclose(
        full_reference_frame["rho"].to_numpy(),
        ordered_inputs_numpy[:, 0]
    )
    and np.allclose(
        full_reference_frame["tau"].to_numpy(),
        ordered_inputs_numpy[:, 1]
    )
):
    raise ValueError(
        "The FVM and Hybrid-PINN grids are not aligned in rho and tau."
    )

corrected_frame = full_reference_frame.rename(
    columns={"concentration": "concentration_reference"}
).copy()
corrected_frame["concentration_pinn"] = ordered_inputs_numpy[:, 2]
corrected_frame["residual_reference"] = (
    corrected_frame["concentration_reference"]
    - corrected_frame["concentration_pinn"]
)
corrected_frame["residual_correction"] = residual_corrections
corrected_frame["concentration_hybrid"] = hybrid_concentrations

corrected_dataset_path.parent.mkdir(parents=True, exist_ok=True)
corrected_frame.to_csv(corrected_dataset_path, index=False)

concentration_reference = corrected_frame[
    "concentration_reference"
].to_numpy()
concentration_pinn = corrected_frame[
    "concentration_pinn"
].to_numpy()
concentration_hybrid = corrected_frame[
    "concentration_hybrid"
].to_numpy()

pinn_comparison_metrics = residual_learner.compute_comparison_metrics(
    corrected_frame,
    "concentration_pinn"
)
hybrid_comparison_metrics = residual_learner.compute_comparison_metrics(
    corrected_frame,
    "concentration_hybrid"
)
physical_checks = residual_learner.compute_physical_checks(
    corrected_frame,
    initial_concentration
)
pinn_relative_l2 = pinn_comparison_metrics["field"]["relative_l2"]
hybrid_relative_l2 = hybrid_comparison_metrics["field"]["relative_l2"]

if pinn_relative_l2 == 0.0:
    error_reduction_percent = 0.0
else:
    error_reduction_percent = 100.0 * (
        1.0 - hybrid_relative_l2 / pinn_relative_l2
    )

hybrid_model = HybridModel(
    pinn_model,
    residual_model
).to(device)
hybrid_model.eval()
hybrid_model.requires_grad_(False)

hard_constraint_checks = (
    residual_learner.compute_hard_constraint_checks(
        hybrid_model=hybrid_model,
        dataset=corrected_frame,
        initial_concentration=initial_concentration,
        device=device
    )
)
initial_condition_max_error = hard_constraint_checks[
    "initial_condition_max_error"
]
centre_symmetry_max_error = hard_constraint_checks[
    "centre_symmetry_max_error"
]
surface_flux_check = residual_learner.compute_surface_flux_check(
    hybrid_model=hybrid_model,
    dataset=corrected_frame,
    initial_concentration=initial_concentration,
    device=device
)
pde_consistency_checks = (
    residual_learner.compute_pde_consistency_checks(
        hybrid_model=hybrid_model,
        dataset=corrected_frame,
        initial_concentration=initial_concentration,
        device=device
    )
)
corrected_frame["full_physics_pde_residual"] = np.nan
corrected_frame["simplified_physics_pde_residual"] = np.nan
corrected_frame.loc[
    pde_consistency_checks["dataset_indices"],
    "full_physics_pde_residual"
] = pde_consistency_checks["full_physics_residual"]
corrected_frame.loc[
    pde_consistency_checks["dataset_indices"],
    "simplified_physics_pde_residual"
] = pde_consistency_checks["simplified_physics_residual"]
corrected_frame.to_csv(corrected_dataset_path, index=False)

total_correction_pipeline_seconds = (
    pinn_inference_seconds
    + residual_training_seconds
    + correction_seconds
)

print("=" * 100)
print("HYBRID CORRECTION TIMING")
print(
    "PINN inference:          "
    f"{pinn_inference_seconds:.6f} s"
)
print(
    "Residual training:       "
    f"{residual_training_seconds:.6f} s"
)
print(
    "Residual correction:     "
    f"{correction_seconds:.6f} s"
)
print(
    "Total correction time:   "
    f"{total_correction_pipeline_seconds:.6f} s"
)
print(
    "Corrected dataset saved: "
    f"{corrected_dataset_path.resolve()}"
)
print("=" * 100)

print("=" * 100)
print("PINN AND HYBRID-PINN FULL-FIELD RESULTS")
print(f"PINN relative L2:       {pinn_relative_l2:.6e}")
print(f"Hybrid relative L2:     {hybrid_relative_l2:.6e}")
print(f"Riduzione dell'errore:  {error_reduction_percent:.2f}%")
print(
    "Range Hybrid:           "
    f"[{np.min(concentration_hybrid):.6f}, "
    f"{np.max(concentration_hybrid):.6f}]"
)
print(
    "Errore IC massimo:      "
    f"{initial_condition_max_error:.6e}"
)
print(
    "Derivata al centro:     "
    f"{centre_symmetry_max_error:.6e}"
)
print("=" * 100)

residual_learner.print_error_metrics("PINN", pinn_comparison_metrics)
residual_learner.print_error_metrics(
    "HYBRID-PINN",
    hybrid_comparison_metrics
)

print("-" * 100)
print("PHYSICAL CHECKS")

for model_name, check_name in (
        ("FVM", "reference"),
        ("PINN", "pinn"),
        ("Hybrid-PINN", "hybrid")
        ):
    model_checks = physical_checks[check_name]
    print(
        f"{model_name:11s} mass balance    | "
        f"{'PASS' if model_checks['mass_balance_passed'] else 'FAIL'} | "
        f"Max |R_M|: {model_checks['mass_balance_max_abs']:.4e} | "
        f"RMSE: {model_checks['mass_balance_rmse']:.4e} | "
        "Final residual: "
        f"{model_checks['mass_balance_final_residual']:.4e} | "
        "Relative mass-change error: "
        f"{model_checks['relative_mass_change_error']:.4e}"
    )
    print(
        f"{model_name:11s} physical bounds | "
        f"{'PASS' if model_checks['physical_bounds_passed'] else 'FAIL'} | "
        f"Range: [{model_checks['minimum_concentration']:.6f}, "
        f"{model_checks['maximum_concentration']:.6f}]"
    )

print(
    "Hybrid hard constraints | "
    f"IC max error: {initial_condition_max_error:.4e} | "
    "Centre derivative max error: "
    f"{centre_symmetry_max_error:.4e}"
)

print("-" * 100)
print("HYBRID-PINN FULL-PHYSICS SURFACE-FLUX CHECK")
print(
    "Surface-flux residual | "
    f"RMSE: {surface_flux_check['rmse']:.4e} | "
    "Max |R_BC|: "
    f"{surface_flux_check['maximum_absolute_error']:.4e} | "
    "Final residual: "
    f"{surface_flux_check['final_residual']:.4e}"
)
print("SURFACE-FLUX RESIDUAL BY OPERATING PHASE")

for phase_name, phase_metrics in surface_flux_check["phases"].items():
    print(
        f"{phase_name:16s} | "
        f"RMSE: {phase_metrics['rmse']:.4e} | "
        "Max |R_BC|: "
        f"{phase_metrics['maximum_absolute_error']:.4e}"
    )

print("-" * 100)
print("HYBRID-PINN FULL-PHYSICS PDE CONSISTENCY")
print(
    "Interior evaluation points: "
    f"{pde_consistency_checks['number_of_points']}"
)
print(
    "Full-physics PDE residual | "
    f"RMSE: {pde_consistency_checks['full_physics']['rmse']:.4e} | "
    "Max |R_PDE|: "
    f"{pde_consistency_checks['full_physics']['maximum_absolute_error']:.4e}"
)
print("FULL-PHYSICS PDE RESIDUAL BY OPERATING PHASE")

for phase_name, phase_metrics in (
        pde_consistency_checks["full_physics"]["phases"].items()
        ):
    print(
        f"{phase_name:16s} | "
        f"RMSE: {phase_metrics['rmse']:.4e} | "
        "Max |R_PDE|: "
        f"{phase_metrics['maximum_absolute_error']:.4e}"
    )

print("-" * 100)
print("HYBRID-PINN SIMPLIFIED-PHYSICS CONSISTENCY")
print(
    "Simplified PDE residual | "
    "RMSE: "
    f"{pde_consistency_checks['simplified_physics']['rmse']:.4e} | "
    "Max absolute residual: "
    f"{pde_consistency_checks['simplified_physics']['maximum_absolute_error']:.4e}"
)
print("SIMPLIFIED-PHYSICS PDE RESIDUAL BY OPERATING PHASE")

for phase_name, phase_metrics in (
        pde_consistency_checks["simplified_physics"]["phases"].items()
        ):
    print(
        f"{phase_name:16s} | "
        f"RMSE: {phase_metrics['rmse']:.4e} | "
        "Max absolute residual: "
        f"{phase_metrics['maximum_absolute_error']:.4e}"
    )

regularity_checks = pde_consistency_checks["regularity"]
print("-" * 100)
print("HYBRID-PINN NUMERICAL REGULARITY")
print(
    "Finite concentration | "
    f"{'PASS' if regularity_checks['concentration_is_finite'] else 'FAIL'}"
)
print(
    "Finite derivatives   | "
    f"{'PASS' if regularity_checks['derivatives_are_finite'] else 'FAIL'}"
)
print(
    "Finite PDE residuals | "
    f"{'PASS' if regularity_checks['pde_residuals_are_finite'] else 'FAIL'}"
)
print(
    "Maximum derivatives  | "
    "|dC/drho|: "
    f"{regularity_checks['maximum_absolute_radial_gradient']:.4e} | "
    "|dC/dtau|: "
    f"{regularity_checks['maximum_absolute_temporal_gradient']:.4e} | "
    "|d2C/drho2|: "
    f"{regularity_checks['maximum_absolute_radial_second_derivative']:.4e}"
)
print("=" * 100)
