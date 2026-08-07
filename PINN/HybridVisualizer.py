import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from Visualizer import PINNVisualizer


class HybridPINNVisualizer(PINNVisualizer):
    """Visualize Hybrid-PINN predictions against the PINN and FVM reference.

    The subclass retains the baseline, correction, and reconstructed field so
    their errors and physical diagnostics can be plotted independently.
    """

    def __init__(
            self,
            num_rho_points=101,
            num_tau_points=201,
            save_im=True,
            images_folder="./Images/Hybrid_PINN",
            comparison_folder="./Images/Comparison"
            ):
        super().__init__(
            num_rho_points=num_rho_points,
            num_tau_points=num_tau_points,
            save_im=save_im,
            images_folder=images_folder,
            comparison_folder=comparison_folder
        )

        self.pinn_concentration = None
        self.residual_correction = None

    def get_hybrid_plot_data(
            self,
            hybrid_model,
            initial_concentration,
            flux,
            tau_final
            ):
        """Evaluate all hybrid components on a structured space--time grid.

        Returns grids followed by PINN, correction, and Hybrid-PINN fields.
        """
        device = next(hybrid_model.parameters()).device

        self.model = hybrid_model
        self.initial_concentration = float(initial_concentration)
        self.flux = float(flux)
        self.radius = np.linspace(
            0.0,
            1.0,
            self.number_rho_points
        )
        self.tau = np.linspace(
            0.0,
            float(tau_final),
            self.number_tau_points
        )

        rho_grid, tau_grid = np.meshgrid(
            self.radius,
            self.tau,
            indexing="ij"
        )
        model_inputs = torch.tensor(
            np.column_stack(
                (rho_grid.ravel(), tau_grid.ravel())
            ),
            dtype=torch.float32,
            device=device
        )

        hybrid_model.eval()

        with torch.inference_mode():
            (
                concentration_pinn,
                residual_correction,
                concentration_hybrid
            ) = hybrid_model.forward_components(
                model_inputs,
                initial_concentration
            )

        field_shape = (
            self.number_rho_points,
            self.number_tau_points
        )
        self.pinn_concentration = (
            concentration_pinn
            .reshape(field_shape)
            .cpu()
            .numpy()
        )
        self.residual_correction = (
            residual_correction
            .reshape(field_shape)
            .cpu()
            .numpy()
        )
        self.concentration = (
            concentration_hybrid
            .reshape(field_shape)
            .cpu()
            .numpy()
        )
        self.time = self.tau * self.diffusion_time

        return (
            self.radius,
            self.tau,
            self.time,
            self.pinn_concentration,
            self.residual_correction,
            self.concentration
        )

    def generate_all_plots(self, n_profiles=5, show=False):
        """Generate and return all standalone-hybrid and comparison figures."""
        self._require_hybrid_data()
        self._require_reference_data()

        figures = {
            "hybrid_radial_profiles": (
                self.plot_hybrid_radial_profiles(n_profiles, show)
            ),
            "hybrid_concentration_map": (
                self.plot_hybrid_concentration_map(show)
            ),
            "hybrid_boundaries": (
                self.plot_hybrid_boundary_concentrations(show)
            ),
            "hybrid_average": (
                self.plot_hybrid_average_concentration(show)
            ),
            "residual_correction": (
                self.plot_residual_correction(show)
            ),
            "comparison_radial_profiles": (
                self.plot_pinn_hybrid_radial_profiles(
                    n_profiles,
                    show
                )
            ),
            "comparison_average": (
                self.plot_pinn_hybrid_average_concentration(show)
            ),
            "comparison_boundaries": (
                self.plot_pinn_hybrid_boundary_concentrations(show)
            ),
            "comparison_fields": (
                self.plot_pinn_hybrid_concentration_fields(show)
            ),
            "comparison_metrics": (
                self.plot_pinn_hybrid_error_metrics(show)
            )
        }

        return figures

    def plot_hybrid_radial_profiles(self, n_profiles=5, show=False):
        indices = self._profile_indices(n_profiles)
        fig, axis = plt.subplots(figsize=(8, 5))

        for index in indices:
            axis.plot(
                self.radius,
                self.concentration[:, index],
                linewidth=2,
                label=f"t = {self.time[index]:.0f} s"
            )

        axis.set_xlabel(r"Dimensionless radius, $\rho$")
        axis.set_ylabel(r"Dimensionless concentration, $C$")
        axis.set_title("Hybrid-PINN radial concentration profiles")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_hybrid_figure(
            fig,
            "Hybrid_PINN_radial_profiles.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_hybrid_concentration_map(self, show=False):
        fig, axis = plt.subplots(figsize=(9, 5))
        mesh = axis.pcolormesh(
            self.time,
            self.radius,
            self.concentration,
            shading="auto"
        )
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel(r"Dimensionless radius, $\rho$")
        axis.set_title("Hybrid-PINN concentration field")
        fig.colorbar(
            mesh,
            ax=axis,
            label="Dimensionless concentration"
        )
        fig.tight_layout()

        self._save_hybrid_figure(
            fig,
            "Hybrid_PINN_concentration_map.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_hybrid_boundary_concentrations(self, show=False):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            self.time,
            self.concentration[0, :],
            linewidth=2,
            label="Particle centre"
        )
        axis.plot(
            self.time,
            self.concentration[-1, :],
            linewidth=2,
            label="Particle surface"
        )
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel("Dimensionless concentration")
        axis.set_title("Hybrid-PINN boundary concentrations")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_hybrid_figure(
            fig,
            "Hybrid_PINN_boundary_concentrations.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_hybrid_average_concentration(self, show=False):
        hybrid_average = self._volume_average(self.concentration)
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(self.time, hybrid_average, linewidth=2)
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel("Average dimensionless concentration")
        axis.set_title("Hybrid-PINN volume-averaged concentration")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_hybrid_figure(
            fig,
            "Hybrid_PINN_average_concentration.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_residual_correction(self, show=False):
        correction_limit = max(
            float(np.max(np.abs(self.residual_correction))),
            np.finfo(float).eps
        )
        fig, axis = plt.subplots(figsize=(9, 5))
        mesh = axis.pcolormesh(
            self.time,
            self.radius,
            self.residual_correction,
            shading="auto",
            cmap="coolwarm",
            vmin=-correction_limit,
            vmax=correction_limit
        )
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel(r"Dimensionless radius, $\rho$")
        axis.set_title(r"Learned residual correction, $\widehat{\Delta C}$")
        fig.colorbar(
            mesh,
            ax=axis,
            label="Concentration correction"
        )
        fig.tight_layout()

        self._save_hybrid_figure(
            fig,
            "Hybrid_PINN_residual_correction_map.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_pinn_hybrid_radial_profiles(
            self,
            n_profiles=5,
            show=False
            ):
        indices = self._profile_indices(n_profiles)
        colors = plt.cm.viridis(
            np.linspace(0.0, 1.0, len(indices))
        )
        fig, axis = plt.subplots(figsize=(9, 5))

        for color, index in zip(colors, indices):
            axis.plot(
                self.radius,
                self.reference_concentration[:, index],
                color=color,
                linewidth=2,
                linestyle="-"
            )
            axis.plot(
                self.radius,
                self.pinn_concentration[:, index],
                color=color,
                linewidth=2,
                linestyle=":"
            )
            axis.plot(
                self.radius,
                self.concentration[:, index],
                color=color,
                linewidth=2,
                linestyle="--"
            )

        time_handles = [
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=f"t = {self.time[index]:.0f} s"
            )
            for color, index in zip(colors, indices)
        ]
        model_handles = [
            Line2D(
                [0], [0], color="black", linewidth=2,
                linestyle="-", label="FVM"
            ),
            Line2D(
                [0], [0], color="black", linewidth=2,
                linestyle=":", label="PINN"
            ),
            Line2D(
                [0], [0], color="black", linewidth=2,
                linestyle="--", label="Hybrid-PINN"
            )
        ]
        time_legend = axis.legend(
            handles=time_handles,
            title="Time",
            loc="upper left"
        )
        axis.add_artist(time_legend)
        axis.legend(handles=model_handles, loc="lower right")
        axis.set_xlabel(r"Dimensionless radius, $\rho$")
        axis.set_ylabel(r"Dimensionless concentration, $C$")
        axis.set_title("FVM, PINN and Hybrid-PINN radial profiles")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(*self._all_model_concentration_limits())
        fig.tight_layout()

        self._save_direct_comparison_figure(
            fig,
            "PINN_Hybrid_radial_profiles_comparison.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_pinn_hybrid_average_concentration(self, show=False):
        reference_average = self._volume_average(
            self.reference_concentration
        )
        pinn_average = self._volume_average(self.pinn_concentration)
        hybrid_average = self._volume_average(self.concentration)

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(self.time, reference_average, linewidth=2, label="FVM")
        axis.plot(
            self.time,
            pinn_average,
            linewidth=2,
            linestyle=":",
            label="PINN"
        )
        axis.plot(
            self.time,
            hybrid_average,
            linewidth=2,
            linestyle="--",
            label="Hybrid-PINN"
        )
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel("Average dimensionless concentration")
        axis.set_title("Volume-averaged concentration comparison")
        axis.set_ylim(*self._all_model_concentration_limits())
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_direct_comparison_figure(
            fig,
            "PINN_Hybrid_average_concentration_comparison.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_pinn_hybrid_boundary_concentrations(self, show=False):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharex=True,
            sharey=True
        )
        boundary_indices = (
            ("Particle centre", 0),
            ("Particle surface", -1)
        )

        for axis, (boundary_name, index) in zip(
                axes,
                boundary_indices
                ):
            axis.plot(
                self.time,
                self.reference_concentration[index, :],
                linewidth=2,
                label="FVM"
            )
            axis.plot(
                self.time,
                self.pinn_concentration[index, :],
                linewidth=2,
                linestyle=":",
                label="PINN"
            )
            axis.plot(
                self.time,
                self.concentration[index, :],
                linewidth=2,
                linestyle="--",
                label="Hybrid-PINN"
            )
            axis.set_title(boundary_name)
            axis.set_xlabel("Physical time [s]")
            axis.grid(True, alpha=0.3)
            axis.legend()

        axes[0].set_ylabel("Dimensionless concentration")
        axes[0].set_ylim(*self._all_model_concentration_limits())
        fig.suptitle("Boundary concentration comparison")
        fig.tight_layout()

        self._save_direct_comparison_figure(
            fig,
            "PINN_Hybrid_boundary_concentrations_comparison.png"
        )
        self._finish_figure(fig, show)

        return fig, axes

    def plot_pinn_hybrid_concentration_fields(self, show=False):
        concentration_minimum = min(
            float(np.min(self.reference_concentration)),
            float(np.min(self.pinn_concentration)),
            float(np.min(self.concentration))
        )
        concentration_maximum = max(
            float(np.max(self.reference_concentration)),
            float(np.max(self.pinn_concentration)),
            float(np.max(self.concentration))
        )
        pinn_error = self.pinn_concentration - self.reference_concentration
        hybrid_error = self.concentration - self.reference_concentration
        error_limit = max(
            float(np.max(np.abs(pinn_error))),
            float(np.max(np.abs(hybrid_error))),
            np.finfo(float).eps
        )
        correction_limit = max(
            float(np.max(np.abs(self.residual_correction))),
            np.finfo(float).eps
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 10),
            sharex=True,
            sharey=True,
            constrained_layout=True
        )
        reference_mesh = axes[0, 0].pcolormesh(
            self.time,
            self.radius,
            self.reference_concentration,
            shading="auto",
            vmin=concentration_minimum,
            vmax=concentration_maximum
        )
        axes[0, 1].pcolormesh(
            self.time,
            self.radius,
            self.pinn_concentration,
            shading="auto",
            vmin=concentration_minimum,
            vmax=concentration_maximum
        )
        axes[0, 2].pcolormesh(
            self.time,
            self.radius,
            self.concentration,
            shading="auto",
            vmin=concentration_minimum,
            vmax=concentration_maximum
        )
        pinn_error_mesh = axes[1, 0].pcolormesh(
            self.time,
            self.radius,
            pinn_error,
            shading="auto",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit
        )
        axes[1, 1].pcolormesh(
            self.time,
            self.radius,
            hybrid_error,
            shading="auto",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit
        )
        correction_mesh = axes[1, 2].pcolormesh(
            self.time,
            self.radius,
            self.residual_correction,
            shading="auto",
            cmap="coolwarm",
            vmin=-correction_limit,
            vmax=correction_limit
        )

        titles = (
            "FVM reference",
            "PINN",
            "Hybrid-PINN",
            r"PINN error, $C_{PINN}-C_{FVM}$",
            r"Hybrid error, $C_{Hybrid}-C_{FVM}$",
            r"Residual correction, $\widehat{\Delta C}$"
        )
        for axis, title in zip(axes.ravel(), titles):
            axis.set_title(title)
            axis.set_xlabel("Physical time [s]")
        axes[0, 0].set_ylabel(r"Dimensionless radius, $\rho$")
        axes[1, 0].set_ylabel(r"Dimensionless radius, $\rho$")

        fig.colorbar(
            reference_mesh,
            ax=axes[0, :],
            label="Dimensionless concentration"
        )
        fig.colorbar(
            pinn_error_mesh,
            ax=axes[1, :2],
            label="Concentration error"
        )
        fig.colorbar(
            correction_mesh,
            ax=axes[1, 2],
            label="Concentration correction"
        )
        fig.suptitle("FVM, PINN and Hybrid-PINN field comparison")

        self._save_direct_comparison_figure(
            fig,
            "PINN_Hybrid_concentration_fields_comparison.png"
        )
        self._finish_figure(fig, show)

        return fig, axes

    def plot_pinn_hybrid_error_metrics(self, show=False):
        pinn_metrics = self._error_metrics(
            self.reference_concentration,
            self.pinn_concentration
        )
        hybrid_metrics = self._error_metrics(
            self.reference_concentration,
            self.concentration
        )
        metric_names = ("relative_l2", "rmse", "mae")
        labels = ("Relative L2", "RMSE", "MAE")
        pinn_values = [pinn_metrics[name] for name in metric_names]
        hybrid_values = [
            hybrid_metrics[name]
            for name in metric_names
        ]
        positions = np.arange(len(metric_names))
        width = 0.36

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            positions - width / 2.0,
            pinn_values,
            width,
            label="PINN"
        )
        axis.bar(
            positions + width / 2.0,
            hybrid_values,
            width,
            label="Hybrid-PINN"
        )
        axis.set_xticks(positions, labels)
        axis.set_yscale("log")
        axis.set_ylabel("Error")
        axis.set_title("PINN and Hybrid-PINN error comparison")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_direct_comparison_figure(
            fig,
            "PINN_Hybrid_error_metrics_comparison.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_full_physics_pde_residual_map(
            self,
            corrected_dataset_path,
            show=False
            ):
        """Plot the stored full-physics PDE residual from a corrected dataset."""
        dataset = pd.read_csv(corrected_dataset_path)
        residual_column = "full_physics_pde_residual"

        if residual_column not in dataset.columns:
            raise ValueError(
                f"{residual_column} is missing from "
                f"{corrected_dataset_path}. Run HybridDemo.py first."
            )

        residual_data = dataset.dropna(subset=[residual_column])

        if residual_data.empty:
            raise ValueError(
                "No interior PDE-residual values were found in the "
                "corrected dataset."
            )

        residual_map = residual_data.pivot(
            index="rho",
            columns="tau",
            values=residual_column
        )
        maximum_absolute_residual = float(
            np.nanmax(np.abs(residual_map.to_numpy()))
        )
        fig, axis = plt.subplots(figsize=(9, 5))
        mesh = axis.pcolormesh(
            residual_map.columns.to_numpy(),
            residual_map.index.to_numpy(),
            residual_map.to_numpy(),
            shading="auto",
            cmap="coolwarm",
            vmin=-maximum_absolute_residual,
            vmax=maximum_absolute_residual
        )
        axis.set_xlabel(r"Dimensionless time, $\tau$")
        axis.set_ylabel(r"Dimensionless radius, $\rho$")
        axis.set_title(
            "Hybrid-PINN full-physics PDE residual"
        )
        fig.colorbar(
            mesh,
            ax=axis,
            label=r"$R_{\mathrm{PDE}}$"
        )
        fig.tight_layout()

        self._save_hybrid_figure(
            fig,
            "Hybrid_PINN_full_physics_PDE_residual.png"
        )
        self._finish_figure(fig, show)

        return fig, axis

    def _volume_average(self, concentration):
        numerator = np.trapezoid(
            concentration * self.radius[:, None]**2,
            self.radius,
            axis=0
        )
        denominator = np.trapezoid(
            self.radius**2,
            self.radius
        )

        return numerator / denominator

    def _profile_indices(self, n_profiles):
        return np.linspace(
            0,
            self.number_tau_points - 1,
            n_profiles,
            dtype=int
        )

    def _all_model_concentration_limits(self):
        concentration_minimum = min(
            float(np.min(self.reference_concentration)),
            float(np.min(self.pinn_concentration)),
            float(np.min(self.concentration))
        )
        concentration_maximum = max(
            float(np.max(self.reference_concentration)),
            float(np.max(self.pinn_concentration)),
            float(np.max(self.concentration))
        )
        concentration_range = concentration_maximum - concentration_minimum
        padding = max(0.05 * concentration_range, 1e-6)

        return (
            concentration_minimum - padding,
            concentration_maximum + padding
        )

    def _require_hybrid_data(self):
        self._require_plot_data()

        if any(
            value is None
            for value in (
                self.pinn_concentration,
                self.residual_correction
            )
        ):
            raise RuntimeError(
                "Call get_hybrid_plot_data() before generating plots."
            )

    def _save_hybrid_figure(self, figure, filename):
        if self.save:
            os.makedirs(self.images_folder, exist_ok=True)
            figure.savefig(
                os.path.join(self.images_folder, filename),
                dpi=300,
                bbox_inches="tight"
            )

    def _save_direct_comparison_figure(self, figure, filename):
        if self.save:
            os.makedirs(self.comparison_folder, exist_ok=True)
            figure.savefig(
                os.path.join(self.comparison_folder, filename),
                dpi=300,
                bbox_inches="tight"
            )

    @staticmethod
    def _finish_figure(figure, show):
        if show:
            plt.show()
        else:
            plt.close(figure)
