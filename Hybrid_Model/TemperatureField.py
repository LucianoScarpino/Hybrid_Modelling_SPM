from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from DatasetLoader import Loader
from ThermalDiscover import FFNDiscovering,SINDyc


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_FOLDER / "Dataset"
THERMAL_VARIANTS_FOLDER = DATASET_FOLDER / "ForThermalVariants"
IMAGES_FOLDER = PROJECT_FOLDER / "Images"

# Manual controls: "SINDYC", "FFN", or "COMPARISON".
MODEL_MODE = "SINDYC"
# "interpolation" or "extrapolation"
TEST_TYPE = "interpolation"                          
SHOW_PLOTS = True

# Common physical and model-selection parameters.
T_AMB = 298.15
THRESHOLD_RANGE = [1e-3,3e-3,1e-2,3e-2]

# FFN training parameters.
FFN_EPOCHS = 2000
FFN_LEARNING_RATE = 1e-3
FFN_EARLY_STOPPING = 200
RANDOM_SEED = 26

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if MODEL_MODE not in {"SINDYC", "FFN", "COMPARISON"}:
    raise ValueError(
        "MODEL_MODE must be 'SINDYC', 'FFN', or 'COMPARISON'."
    )

if TEST_TYPE not in {"interpolation", "extrapolation"}:
    raise ValueError(
        "TEST_TYPE must be 'interpolation' or 'extrapolation'."
    )

dataset_names = [
    f"full_simulation_dataset_{index}.csv"
    for index in range(8)
]

thermal_datasets = [
    pd.read_csv(THERMAL_VARIANTS_FOLDER / dataset_name)
    for dataset_name in dataset_names
]

training_datasets = thermal_datasets[0:5]
validation_dataset = thermal_datasets[5]
test_dataset_index = 6 if TEST_TYPE == "interpolation" else 7
test_dataset = thermal_datasets[test_dataset_index]

device = Loader().get_device()
IMAGES_FOLDER.mkdir(parents=True,exist_ok=True)

results = {}
training_times = {}
model_complexities = {}

if MODEL_MODE in {"SINDYC", "COMPARISON"}:
    sindyc = SINDyc(training_datasets)

    sindyc_training_start = perf_counter()
    sindyc_model,_,selected_threshold = sindyc.discover_equation(
        THRESHOLD_RANGE,
        validation_dataset=validation_dataset,
        T_amb=T_AMB
    )
    training_times["SINDYC"] = perf_counter() - sindyc_training_start

    results["SINDYC"] = sindyc.evaluate_dataset(
        sindyc_model,
        test_dataset,
        T_amb=T_AMB
    )
    model_complexities["SINDYC"] = sindyc_model.complexity

if MODEL_MODE in {"FFN", "COMPARISON"}:
    ffn = FFNDiscovering(
        THERMAL_VARIANTS_FOLDER,
        dataset_names,
        T_amb=T_AMB
    )
    (
        train_loader,
        validation_loader,
        interpolation_test_loader,
        extrapolation_test_loader,
    ) = ffn.generate_thermal_loaders()

    selected_test_loader = (
        interpolation_test_loader
        if TEST_TYPE == "interpolation"
        else extrapolation_test_loader
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    ffn_training_start = perf_counter()
    ffn_model = ffn.predict_equations(
        device=device,
        lr=FFN_LEARNING_RATE,
        training_epochs=FFN_EPOCHS,
        early_stopping_epochs=FFN_EARLY_STOPPING,
        train_loader=train_loader,
        val_loader=validation_loader
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    training_times["FFN"] = perf_counter() - ffn_training_start

    results["FFN"] = ffn.test_thermalFNN(
        ffn_model,
        selected_test_loader,
        device,
        test_type=TEST_TYPE,
        T_amb=T_AMB
    )
    model_complexities["FFN"] = sum(
        parameter.numel()
        for parameter in ffn_model.parameters()
    )

plot_filenames = {
    "SINDYC": f"SINDYc_temperature_map_{TEST_TYPE}.png",
    "FFN": f"FFN_temperature_map_{TEST_TYPE}.png",
}

for model_name,model_results in results.items():
    reference_temperature = model_results["reference_temperature"]
    predicted_temperature = model_results["predicted_temperature"]

    figure,axis = plt.subplots(figsize=(9,5))
    mesh = axis.pcolormesh(
        model_results["time"],
        model_results["radius"],
        model_results["predicted_temperature_map"],
        shading="auto",
        vmin=min(reference_temperature.min(),predicted_temperature.min()),
        vmax=max(reference_temperature.max(),predicted_temperature.max())
    )

    axis.set_xlabel("Physical time [s]")
    axis.set_ylabel("Dimensionless radius, $\\rho$")
    axis.set_title(
        f"{model_name} temperature field ({TEST_TYPE} test)"
    )

    colorbar = figure.colorbar(mesh,ax=axis)
    colorbar.set_label("Temperature [K]")
    colorbar.formatter.set_useOffset(False)
    colorbar.formatter.set_scientific(False)
    colorbar.update_ticks()

    figure.tight_layout()
    figure.savefig(
        IMAGES_FOLDER / plot_filenames[model_name],
        dpi=300,
        bbox_inches="tight"
    )

    if not SHOW_PLOTS:
        plt.close(figure)

if MODEL_MODE == "COMPARISON":
    comparison_figure,comparison_axis = plt.subplots(figsize=(9,5))
    comparison_axis.plot(
        results["SINDYC"]["time"],
        results["SINDYC"]["reference_temperature"],
        color="black",
        linewidth=2.0,
        label="FVM reference"
    )
    comparison_axis.plot(
        results["SINDYC"]["time"],
        results["SINDYC"]["predicted_temperature"],
        linestyle="--",
        linewidth=1.8,
        label="SINDYc"
    )
    comparison_axis.plot(
        results["FFN"]["time"],
        results["FFN"]["predicted_temperature"],
        linestyle=":",
        linewidth=2.0,
        label="FFN"
    )
    comparison_axis.set_xlabel("Physical time [s]")
    comparison_axis.set_ylabel("Temperature [K]")
    comparison_axis.set_title(
        f"Thermal-model comparison ({TEST_TYPE} test)"
    )
    comparison_axis.grid(True,alpha=0.25)
    comparison_axis.legend()
    comparison_figure.tight_layout()
    comparison_figure.savefig(
        IMAGES_FOLDER / f"Thermal_models_comparison_{TEST_TYPE}.png",
        dpi=300,
        bbox_inches="tight"
    )

    if not SHOW_PLOTS:
        plt.close(comparison_figure)

metric_labels = {
    "temperature_rmse": "Temperature RMSE [K]",
    "temperature_mae": "Temperature MAE [K]",
    "temperature_max_error": "Maximum error [K]",
    "relative_l2_theta": "Relative L2 on theta",
    "peak_temperature_error": "Peak-temperature error [K]",
    "peak_time_error": "Peak-time error [s]",
    "peak_current_rmse": "Peak-current RMSE [K]",
    "cooling_rmse": "Cooling RMSE [K]",
    "final_temperature_error": "Final-temperature error [K]",
    "derivative_rmse": "Derivative RMSE [K/s]",
    "derivative_mae": "Derivative MAE [K/s]",
}

print("=" * 108)
print(f"THERMAL MODEL RESULTS | test type: {TEST_TYPE} | device: {device}")
print("=" * 108)

if MODEL_MODE == "COMPARISON":
    print(f"{'Metric':42s} | {'SINDYc':>20s} | {'FFN':>20s}")
    print("-" * 108)
    for metric_name,label in metric_labels.items():
        print(
            f"{label:42s} | "
            f"{results['SINDYC']['metrics'][metric_name]:20.6e} | "
            f"{results['FFN']['metrics'][metric_name]:20.6e}"
        )
    print("-" * 108)
    print(
        f"{'Training/identification time [s]':42s} | "
        f"{training_times['SINDYC']:20.6e} | "
        f"{training_times['FFN']:20.6e}"
    )
    print(
        f"{'Rollout inference time [s]':42s} | "
        f"{results['SINDYC']['metrics']['rollout_seconds']:20.6e} | "
        f"{results['FFN']['metrics']['rollout_seconds']:20.6e}"
    )
    print(
        f"{'Active terms / trainable parameters':42s} | "
        f"{model_complexities['SINDYC']:20d} | "
        f"{model_complexities['FFN']:20d}"
    )
else:
    selected_model = MODEL_MODE
    for metric_name,label in metric_labels.items():
        print(
            f"{label:42s} | "
            f"{results[selected_model]['metrics'][metric_name]:.6e}"
        )
    print(
        f"{'Training/identification time [s]':42s} | "
        f"{training_times[selected_model]:.6e}"
    )
    print(
        f"{'Rollout inference time [s]':42s} | "
        f"{results[selected_model]['metrics']['rollout_seconds']:.6e}"
    )
    print(
        f"{'Active terms / trainable parameters':42s} | "
        f"{model_complexities[selected_model]}"
    )

print("=" * 108)

if SHOW_PLOTS:
    plt.show()
