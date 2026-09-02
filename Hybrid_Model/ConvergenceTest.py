from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from DatasetLoader import Loader
from NeuralNetwork import FNN
from Tester import Testing


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_FOLDER / "Dataset"
VARIANTS_FOLDER = DATASET_FOLDER / "Variants"
RESULTS_FOLDER = PROJECT_FOLDER / "Results"
MODELS_FOLDER = RESULTS_FOLDER / "Models"
IMAGES_FOLDER = PROJECT_FOLDER / "Images" / "Convergence"
VARIANT_NUMBERS = range(5)
SHOW_IMAGES = True


def newest_selected_checkpoint(models_folder):
    """Return the most recently saved selected checkpoint."""
    checkpoint_paths = list(
        models_folder.glob("selected_checkpoint_*.pth")
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            "No selected checkpoint was found. Run HybridDemo.py first."
        )

    return max(
        checkpoint_paths,
        key=lambda checkpoint_path: checkpoint_path.stat().st_mtime
    )


def plot_convergence_metrics(variant_numbers,variant_metrics):
    """Plot the error metrics obtained for every simplified-model variant."""
    figure,axes = plt.subplots(
        2,
        2,
        figsize=(11,8),
        sharex=True,
        constrained_layout=True
    )

    comparisons = (
        (
            "Concentration RMSE",
            "baseline_concentration_rmse",
            "hybrid_concentration_rmse"
        ),
        (
            "Surface concentration RMSE",
            "baseline_surface_concentration_rmse",
            "hybrid_surface_concentration_rmse"
        ),
        (
            "Average concentration RMSE",
            "baseline_average_concentration_rmse",
            "hybrid_average_concentration_rmse"
        )
    )

    for axis,(title,baseline_key,hybrid_key) in zip(
            axes.ravel()[:3],
            comparisons
            ):
        axis.plot(
            variant_numbers,
            [metrics[baseline_key] for metrics in variant_metrics],
            marker="o",
            label="Simplified"
        )
        axis.plot(
            variant_numbers,
            [metrics[hybrid_key] for metrics in variant_metrics],
            marker="o",
            label="Hybrid"
        )
        axis.set_title(title)
        axis.set_yscale("log")
        axis.grid(True,which="both",alpha=0.3)
        axis.legend()

    axes[1,1].plot(
        variant_numbers,
        [
            metrics["concentration_rmse_reduction_percent"]
            for metrics in variant_metrics
        ],
        marker="o",
        color="tab:green"
    )
    axes[1,1].axhline(0.0,color="black",linestyle="--")
    axes[1,1].set_title("Concentration RMSE reduction")
    axes[1,1].set_ylabel("RER [%]")
    axes[1,1].grid(True,alpha=0.3)

    for axis in axes[1,:]:
        axis.set_xlabel("Variant number")

    figure.suptitle("Hybrid convergence test")
    IMAGES_FOLDER.mkdir(parents=True,exist_ok=True)
    figure.savefig(
        IMAGES_FOLDER / "convergence_test.png",
        dpi=200,
        bbox_inches="tight"
    )

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close(figure)


device = Loader().get_device()
checkpoint_path = newest_selected_checkpoint(MODELS_FOLDER)
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

if checkpoint.get("constraint") != "constant_volume_average":
    raise ValueError(
        "The selected checkpoint does not use the constant-volume-average "
        "constraint. Run HybridDemo.py again."
    )

inventory_offset = float(checkpoint["inventory_offset"])

trained_model = FNN(
    input_dim=3,
    hidden_dim=32,
    output_dim=101
).to(device)
trained_model.load_state_dict(checkpoint["model_state_dict"])
trained_model.eval()

print(f"Selected checkpoint: {checkpoint_path.resolve()}")
print(f"Inventory offset: {inventory_offset:.4e}")

variant_metrics = []

for number in VARIANT_NUMBERS:
    print(f"Testing simplified-model variant {number}...")

    dataloader = Loader()
    simplified_dataset = dataloader.get_simplified_dataset(
        DATASET_FOLDER,
        variant=True,
        number=number
    )
    reference_dataset = dataloader.get_reference_dataset(DATASET_FOLDER)

    (X_train,y_train),(X_val,y_val),(X_test,y_test) = (
        dataloader.split_dataset(reference_dataset,simplified_dataset)
    )

    # Use the complete variant as an out-of-distribution test trajectory.
    X_test = torch.cat((X_train,X_val,X_test),dim=0)
    y_test = torch.cat((y_train,y_val,y_test),dim=0)
    simplified_profiles = torch.cat(
        (
            dataloader.get_simplified_profiles("train"),
            dataloader.get_simplified_profiles("validation"),
            dataloader.get_simplified_profiles("test")
        ),
        dim=0
    )
    test_tau = np.concatenate(
        (
            dataloader.get_tau("train"),
            dataloader.get_tau("validation"),
            dataloader.get_tau("test")
        )
    )

    test_loader = dataloader.loader(X_test,y_test)
    tester = Testing(
        test_loader,
        dataloader.radius,
        simplified_profiles,
        test_tau,
        inventory_offset
    )

    tester.test(
        trained_model,
        device,
        training_time_seconds=0.0,
        maximum_training_epochs=0,
        completed_training_epochs=0,
        ending_training_epoch=0,
        early_stopped=False,
        output_folder=RESULTS_FOLDER,
        filename="test_history_variants.csv",
        variant_number=number
    )
    variant_metrics.append(tester.concentration_metrics)

    simplified_concentrations,correction = tester.compute_corrections(
        trained_model,
        device,
        simplified_dataset
    )
    corrected_concentrations = simplified_concentrations + correction

    tester.generate_new_dataset(
        simplified_dataset,
        corrected_concentrations,
        VARIANTS_FOLDER,
        filename=f"corrected_dataset_{number}.csv"
    )

plot_convergence_metrics(list(VARIANT_NUMBERS),variant_metrics)
