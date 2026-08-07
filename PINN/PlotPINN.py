import argparse

import torch

from pathlib import Path

from HybridVisualizer import HybridPINNVisualizer
from NeuralNetwork import FFN, HybridModel, MLP
from Visualizer import PINNVisualizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate PINN plots, Hybrid-PINN plots, or both."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("pinn", "hybrid", "all"),
        default="all",
        help=(
            "pinn: regenerate only PINN/FVM plots; "
            "hybrid: regenerate only Hybrid-PINN and direct comparison "
            "plots; all: regenerate both sets (default)."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display the generated figures on screen in addition to "
            "saving them. By default figures are only saved."
        )
    )

    return parser.parse_args()


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_pinn_model(checkpoint_path, device):
    """Load a trained PINN checkpoint and return model plus metadata."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model = MLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def find_latest_residual_checkpoint(models_folder):
    """Return the newest timestamped residual-model checkpoint."""
    candidates = list(
        models_folder.glob("residual_model_*.pth")
    )

    if not candidates:
        raise FileNotFoundError(
            "No residual_model_*.pth checkpoint was found in "
            f"{models_folder}."
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime
    )


def load_residual_model(checkpoint_path, device):
    """Load and freeze a residual FFN from ``checkpoint_path``."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model = FFN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate_pinn_plots(
        model,
        reference_dataset_path,
        initial_concentration,
        flux,
        tau_final,
        num_rho_points,
        num_tau_points,
        show
        ):
    """Generate standalone PINN plots and FVM comparison diagnostics."""
    visualizer = PINNVisualizer(
        num_rho_points,
        num_tau_points,
        save_im=True
    )
    visualizer.get_plot_data(
        model=model,
        initial_concentration=initial_concentration,
        tau_final=tau_final,
        flux=flux
    )
    visualizer.load_reference_data(reference_dataset_path)

    visualizer.show_all(animate=False, show=show)
    visualizer.show_comparison(show=show)

    print("PINN plots saved in: Images/PINN")
    print("FVM-PINN comparison plots saved in: Images/Comparison")


def generate_hybrid_plots(
        pinn_model,
        residual_model,
        residual_checkpoint_path,
        reference_dataset_path,
        corrected_dataset_path,
        initial_concentration,
        flux,
        tau_final,
        num_rho_points,
        num_tau_points,
        device,
        show
        ):
    """Generate Hybrid-PINN plots without retraining either component."""
    hybrid_model = HybridModel(
        pinn_model,
        residual_model
    ).to(device)
    hybrid_model.eval()
    hybrid_model.requires_grad_(False)

    visualizer = HybridPINNVisualizer(
        num_rho_points=num_rho_points,
        num_tau_points=num_tau_points,
        save_im=True,
        images_folder="./Images/Hybrid_PINN",
        comparison_folder="./Images/Comparison"
    )
    visualizer.get_hybrid_plot_data(
        hybrid_model=hybrid_model,
        initial_concentration=initial_concentration,
        flux=flux,
        tau_final=tau_final
    )
    visualizer.load_reference_data(reference_dataset_path)
    visualizer.generate_all_plots(show=show)
    visualizer.plot_full_physics_pde_residual_map(
        corrected_dataset_path=corrected_dataset_path,
        show=show
    )

    print(f"Residual checkpoint:     {residual_checkpoint_path}")
    print("Hybrid-PINN plots saved in: Images/Hybrid_PINN")
    print("Comparison plots saved in: Images/Comparison")


def main():
    arguments = parse_arguments()
    device = select_device()

    models_folder = Path("./Results/Models")
    pinn_checkpoint_path = (
        models_folder
        / "pinn_checkpoint_20260731_120647_711512.pth"
    )
    reference_dataset_path = Path(
        "./Data/pinn_training_dataset.csv"
    )
    corrected_dataset_path = Path(
        "./Data/corrected_dataset.csv"
    )

    pinn_model, pinn_checkpoint = load_pinn_model(
        pinn_checkpoint_path,
        device
    )

    initial_concentration = pinn_checkpoint["initial_concentration"]
    flux = pinn_checkpoint["flux"]
    tau_final = 1.0073356401384084
    num_tau_points = 201
    num_rho_points = 101

    print(f"Plot mode: {arguments.mode}")

    if arguments.mode in ("pinn", "all"):
        generate_pinn_plots(
            model=pinn_model,
            reference_dataset_path=reference_dataset_path,
            initial_concentration=initial_concentration,
            flux=flux,
            tau_final=tau_final,
            num_rho_points=num_rho_points,
            num_tau_points=num_tau_points,
            show=arguments.show
        )

    if arguments.mode in ("hybrid", "all"):
        residual_checkpoint_path = find_latest_residual_checkpoint(
            models_folder
        )
        residual_model = load_residual_model(
            residual_checkpoint_path,
            device
        )
        generate_hybrid_plots(
            pinn_model=pinn_model,
            residual_model=residual_model,
            residual_checkpoint_path=residual_checkpoint_path,
            reference_dataset_path=reference_dataset_path,
            corrected_dataset_path=corrected_dataset_path,
            initial_concentration=initial_concentration,
            flux=flux,
            tau_final=tau_final,
            num_rho_points=num_rho_points,
            num_tau_points=num_tau_points,
            device=device,
            show=arguments.show
        )


if __name__ == "__main__":
    main()
