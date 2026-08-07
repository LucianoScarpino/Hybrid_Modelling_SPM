from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.integrate import cumulative_trapezoid


PROJECT_FOLDER = Path(__file__).resolve().parent
SOFTWARE_FOLDER = PROJECT_FOLDER.parent
PINN_FOLDER = SOFTWARE_FOLDER / "PINN"

DATASET_FOLDER = PROJECT_FOLDER / "Dataset"
IMAGES_FOLDER = PROJECT_FOLDER / "Images" / "Comparison"
RESULTS_FOLDER = PROJECT_FOLDER / "Results"

REFERENCE_DATASET = DATASET_FOLDER / "full_simulation_dataset.csv"
SIMPLIFIED_DATASET = DATASET_FOLDER / "simplified_simulation_dataset.csv"
HYBRID_DATASET = DATASET_FOLDER / "corrected_dataset.csv"

PINN_CHECKPOINT = (
    PINN_FOLDER
    / "Results"
    / "Models"
    / "pinn_checkpoint_20260731_120647_711512.pth"
)
PINN_NETWORK = PINN_FOLDER / "NeuralNetwork.py"

SHOW_IMAGES = True


def load_concentration_field(dataset_path):
    """Load a CSV and return its structured concentration field and states."""
    dataset = pd.read_csv(dataset_path)

    concentration_field = (
        dataset
        .pivot(index="rho",columns="tau",values="concentration")
        .sort_index()
        .sort_index(axis=1)
    )

    states = (
        dataset
        .drop_duplicates("tau")
        .set_index("tau")
        .sort_index()
        .loc[concentration_field.columns]
    )

    return {
        "dataset": dataset,
        "radius": concentration_field.index.to_numpy(dtype=float),
        "tau": concentration_field.columns.to_numpy(dtype=float),
        "time": states["time"].to_numpy(dtype=float),
        "concentration": concentration_field.to_numpy(dtype=float),
        "states": states
    }


def check_common_grid(reference,*models):
    """Raise when any model does not share the reference space-time grid."""
    for model in models:
        if not np.allclose(reference["radius"],model["radius"]):
            raise ValueError("The radial grids do not match.")

        if not np.allclose(reference["tau"],model["tau"]):
            raise ValueError("The time grids do not match.")


def load_pinn_model():
    """Load the frozen PINN and return model, checkpoint metadata, and device."""
    network_spec = spec_from_file_location(
        "pinn_neural_network",
        PINN_NETWORK
    )

    if network_spec is None or network_spec.loader is None:
        raise ImportError("The PINN network module could not be loaded.")

    network_module = module_from_spec(network_spec)
    network_spec.loader.exec_module(network_module)

    device = torch.device("cpu")
    checkpoint = torch.load(
        PINN_CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model = network_module.MLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model,checkpoint,device


def compute_pinn_field(radius,tau):
    """Evaluate the frozen PINN on ``radius`` x ``tau`` and return field/flux."""
    model,checkpoint,device = load_pinn_model()

    rho_grid,tau_grid = np.meshgrid(
        radius,
        tau,
        indexing="ij"
    )

    model_inputs = torch.tensor(
        np.column_stack((rho_grid.ravel(),tau_grid.ravel())),
        dtype=torch.float32,
        device=device
    )

    with torch.inference_mode():
        concentration = model(
            model_inputs,
            checkpoint["initial_concentration"]
        )

    concentration = (
        concentration
        .reshape(len(radius),len(tau))
        .detach()
        .cpu()
        .numpy()
    )

    return concentration,float(checkpoint["flux"])


def compute_average(concentration,radius):
    numerator = np.trapezoid(
        concentration * radius[:,None] ** 2,
        radius,
        axis=0
    )
    denominator = np.trapezoid(radius ** 2,radius)

    return numerator / denominator


def compute_error_metrics(reference,prediction):
    """Return relative L2, RMSE, MAE, and maximum error for two arrays."""
    error = prediction - reference
    reference_norm = np.linalg.norm(reference.ravel())

    return {
        "relative_l2": float(
            np.linalg.norm(error.ravel()) / reference_norm
        ),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error)))
    }


def compute_mass_balance_residual(
        concentration,
        radius,
        tau,
        flux_profile
        ):
    """Return the global mass-balance residual over the supplied time grid."""

    average_concentration = compute_average(concentration,radius)
    expected_average = (
        average_concentration[0]
        - 3.0 * cumulative_trapezoid(
            flux_profile,
            tau,
            initial=0.0
        )
    )

    return average_concentration - expected_average


def save_figure(fig,filename):
    IMAGES_FOLDER.mkdir(parents=True,exist_ok=True)
    fig.savefig(IMAGES_FOLDER / filename,dpi=200,bbox_inches="tight")

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close(fig)


def plot_concentration_fields(time,radius,fields):
    selected_fields = (
        fields["Reference"],
        fields["PINN"],
        fields["Hybrid"]
    )
    color_minimum = min(np.min(field) for field in selected_fields)
    color_maximum = max(np.max(field) for field in selected_fields)

    fig,axes = plt.subplots(
        1,
        3,
        figsize=(15,4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    mesh = None

    for axis,(name,field) in zip(
            axes,
            (
                ("Reference FVM",fields["Reference"]),
                ("PINN",fields["PINN"]),
                ("Hybrid",fields["Hybrid"])
            )
            ):

        mesh = axis.pcolormesh(
            time,
            radius,
            field,
            shading="auto",
            vmin=color_minimum,
            vmax=color_maximum
        )
        axis.set_title(name)
        axis.set_xlabel("Physical time [s]")

    axes[0].set_ylabel("Dimensionless radius, $\\rho$")
    fig.colorbar(
        mesh,
        ax=axes,
        label="Dimensionless concentration"
    )
    fig.suptitle("Concentration field comparison")

    save_figure(fig,"concentration_fields_comparison.png")


def plot_average_concentrations(time,radius,fields):
    fig,axis = plt.subplots(figsize=(9,5))

    styles = {
        "Reference": {"linewidth": 2.5,"color": "black"},
        "Simplified": {"linewidth": 1.8,"linestyle": ":"},
        "PINN": {"linewidth": 1.8,"linestyle": "--"},
        "Hybrid": {"linewidth": 1.8,"linestyle": "-."}
    }

    for name,field in fields.items():
        axis.plot(
            time,
            compute_average(field,radius),
            label=name,
            **styles[name]
        )

    axis.set_xlabel("Physical time [s]")
    axis.set_ylabel("Average dimensionless concentration")
    axis.set_title("Volume-averaged concentration comparison")
    axis.grid(True,alpha=0.3)
    axis.legend()
    fig.tight_layout()

    save_figure(fig,"average_concentration_comparison.png")


def plot_boundary_concentrations(time,fields):
    fig,axes = plt.subplots(
        2,
        1,
        figsize=(9,8),
        sharex=True,
        constrained_layout=True
    )

    styles = {
        "Reference": {"linewidth": 2.5,"color": "black"},
        "Simplified": {"linewidth": 1.8,"linestyle": ":"},
        "PINN": {"linewidth": 1.8,"linestyle": "--"},
        "Hybrid": {"linewidth": 1.8,"linestyle": "-."}
    }

    for name,field in fields.items():
        axes[0].plot(
            time,
            field[0,:],
            label=name,
            **styles[name]
        )
        axes[1].plot(
            time,
            field[-1,:],
            label=name,
            **styles[name]
        )

    axes[0].set_title("Particle centre")
    axes[1].set_title("Particle surface")
    axes[1].set_xlabel("Physical time [s]")

    for axis in axes:
        axis.set_ylabel("Dimensionless concentration")
        axis.grid(True,alpha=0.3)

    axes[0].legend(ncol=2)
    fig.suptitle("Boundary concentration comparison")

    save_figure(fig,"boundary_concentrations_comparison.png")


def plot_radial_profiles(time,radius,fields,n_profiles=5):
    indices = np.linspace(0,len(time) - 1,n_profiles,dtype=int)
    fig,axes = plt.subplots(
        2,
        3,
        figsize=(14,8),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    axes = axes.ravel()

    styles = {
        "Reference": {"linewidth": 2.5,"color": "black"},
        "Simplified": {"linewidth": 1.8,"linestyle": ":"},
        "PINN": {"linewidth": 1.8,"linestyle": "--"},
        "Hybrid": {"linewidth": 1.8,"linestyle": "-."}
    }

    for axis,index in zip(axes,indices):
        for name,field in fields.items():
            axis.plot(
                radius,
                field[:,index],
                label=name,
                **styles[name]
            )

        axis.set_title(f"t = {time[index]:.0f} s")
        axis.grid(True,alpha=0.3)

    axes[-1].axis("off")
    axes[0].legend()

    for axis in axes[:3]:
        axis.set_ylabel("Dimensionless concentration")

    for axis in axes[3:5]:
        axis.set_xlabel("Dimensionless radius, $\\rho$")
        axis.set_ylabel("Dimensionless concentration")

    fig.suptitle("Radial concentration profile comparison")

    save_figure(fig,"radial_profiles_comparison.png")


def plot_mass_balance_residuals(time,residuals):
    fig,axes = plt.subplots(
        2,
        1,
        figsize=(9,8),
        sharex=True,
        constrained_layout=True
    )

    for name,residual in residuals.items():
        axes[0].plot(time,residual,linewidth=1.8,label=name)

    for name in ("Reference","Simplified","Hybrid"):
        axes[1].plot(
            time,
            residuals[name],
            linewidth=1.8,
            label=name
        )

    axes[0].set_title("Complete scale")
    axes[1].set_title("Detail close to zero")
    axes[1].set_xlabel("Physical time [s]")

    for axis in axes:
        axis.axhline(0.0,color="black",linestyle="--",linewidth=1)
        axis.set_ylabel("Mass balance residual")
        axis.grid(True,alpha=0.3)
        axis.legend()

    fig.suptitle("Mass balance comparison")

    save_figure(fig,"mass_balance_comparison.png")


def plot_error_metrics(metrics):
    quantities = ("field","centre","surface","average")
    methods = ("Simplified","PINN","Hybrid")
    x_coordinates = np.arange(len(quantities))
    width = 0.25

    fig,axis = plt.subplots(figsize=(10,5))

    for method_index,method in enumerate(methods):
        rmse_values = [
            metrics[method][quantity]["rmse"]
            for quantity in quantities
        ]
        axis.bar(
            x_coordinates + (method_index - 1) * width,
            rmse_values,
            width,
            label=method
        )

    axis.set_yscale("log")
    axis.set_xticks(x_coordinates,quantities)
    axis.set_ylabel("RMSE")
    axis.set_title("Error with respect to the reference FVM solution")
    axis.grid(True,axis="y",which="both",alpha=0.3)
    axis.legend()
    fig.tight_layout()

    save_figure(fig,"error_metrics_comparison.png")


def save_metrics(metrics,residuals):
    """Serialize cross-model error and mass-balance summaries to CSV."""
    rows = []

    for method,method_metrics in metrics.items():
        for quantity,quantity_metrics in method_metrics.items():
            rows.append({
                "method": method,
                "quantity": quantity,
                **quantity_metrics
            })

    metrics_frame = pd.DataFrame(rows)
    maximum_residuals = {
        method: float(np.max(np.abs(residual)))
        for method,residual in residuals.items()
    }
    metrics_frame["maximum_mass_balance_residual"] = (
        metrics_frame["method"].map(maximum_residuals)
    )

    RESULTS_FOLDER.mkdir(parents=True,exist_ok=True)
    metrics_frame.to_csv(
        RESULTS_FOLDER / "method_comparison_metrics.csv",
        index=False
    )


def print_metrics(metrics,residuals):
    print("=" * 100)
    print("METHOD COMPARISON WITH THE REFERENCE FVM SOLUTION")

    for method in ("Simplified","PINN","Hybrid"):
        print("-" * 100)
        print(method)

        for quantity in ("field","centre","surface","average"):
            quantity_metrics = metrics[method][quantity]
            print(
                f"{quantity.capitalize():8s} | "
                f"Relative L2: {quantity_metrics['relative_l2']:.4e} | "
                f"RMSE: {quantity_metrics['rmse']:.4e} | "
                f"MAE: {quantity_metrics['mae']:.4e} | "
                "Maximum absolute error: "
                f"{quantity_metrics['maximum_absolute_error']:.4e}"
            )

        print(
            "Maximum mass balance residual: "
            f"{np.max(np.abs(residuals[method])):.4e}"
        )

    print("=" * 100)


def main():
    reference = load_concentration_field(REFERENCE_DATASET)
    simplified = load_concentration_field(SIMPLIFIED_DATASET)
    hybrid = load_concentration_field(HYBRID_DATASET)

    check_common_grid(reference,simplified,hybrid)

    pinn_concentration,pinn_flux = compute_pinn_field(
        reference["radius"],
        reference["tau"]
    )

    fields = {
        "Reference": reference["concentration"],
        "Simplified": simplified["concentration"],
        "PINN": pinn_concentration,
        "Hybrid": hybrid["concentration"]
    }

    reference_average = compute_average(
        fields["Reference"],
        reference["radius"]
    )

    metrics = {}

    for method in ("Simplified","PINN","Hybrid"):
        method_average = compute_average(
            fields[method],
            reference["radius"]
        )
        metrics[method] = {
            "field": compute_error_metrics(
                fields["Reference"],
                fields[method]
            ),
            "centre": compute_error_metrics(
                fields["Reference"][0,:],
                fields[method][0,:]
            ),
            "surface": compute_error_metrics(
                fields["Reference"][-1,:],
                fields[method][-1,:]
            ),
            "average": compute_error_metrics(
                reference_average,
                method_average
            )
        }

    reference_flux = reference["states"][
        "dimensionless_flux"
    ].to_numpy(dtype=float)
    # The simplified-model current column stores the dimensional current.
    # The reference dimensionless flux is therefore used for all models that
    # follow the operating current profile. The PINN instead retains the
    # constant flux imposed during its original training.
    residuals = {
        "Reference": compute_mass_balance_residual(
            fields["Reference"],
            reference["radius"],
            reference["tau"],
            reference_flux
        ),
        "Simplified": compute_mass_balance_residual(
            fields["Simplified"],
            reference["radius"],
            reference["tau"],
            reference_flux
        ),
        "PINN": compute_mass_balance_residual(
            fields["PINN"],
            reference["radius"],
            reference["tau"],
            np.full_like(reference["tau"],pinn_flux)
        ),
        "Hybrid": compute_mass_balance_residual(
            fields["Hybrid"],
            reference["radius"],
            reference["tau"],
            reference_flux
        )
    }

    plot_concentration_fields(
        reference["time"],
        reference["radius"],
        fields
    )
    plot_average_concentrations(
        reference["time"],
        reference["radius"],
        fields
    )
    plot_boundary_concentrations(reference["time"],fields)
    plot_radial_profiles(
        reference["time"],
        reference["radius"],
        fields
    )
    plot_mass_balance_residuals(reference["time"],residuals)
    plot_error_metrics(metrics)

    save_metrics(metrics,residuals)
    print_metrics(metrics,residuals)


if __name__ == "__main__":
    main()
