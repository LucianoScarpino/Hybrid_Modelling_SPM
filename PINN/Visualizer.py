import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import erfc


class PINNVisualizer(object):
    """Generate standalone PINN plots and FVM comparison diagnostics.

    The class evaluates a trained model on a structured grid, aligns reference
    data, computes physical/error summaries, and saves publication-ready plots.
    """

    def __init__(
            self,
            num_rho_points=101,
            num_tau_points=201,
            save_im=True,
            images_folder="./Images/PINN",
            comparison_folder="./Images/Comparison",
            ramp_time=300.0,
            ramp_down_start=3600.0
            ):
        self.number_rho_points = num_rho_points
        self.number_tau_points = num_tau_points

        self.particle_radius = 8.5e-6
        self.reference_diffusivity = 1.213e-14
        self.diffusion_time = (
            self.particle_radius**2
            / self.reference_diffusivity
            )

        self.save = save_im
        self.images_folder = images_folder
        self.comparison_folder = comparison_folder
        os.makedirs(
            self.images_folder,
            exist_ok=True
            )
        os.makedirs(
            self.comparison_folder,
            exist_ok=True
            )

        self.ramp_time = float(ramp_time)
        self.ramp_down_start = float(ramp_down_start)

        self.model = None
        self.initial_concentration = None
        self.flux = None
        self.radius = None
        self.tau = None
        self.time = None
        self.concentration = None

        self.reference_radius = None
        self.reference_tau = None
        self.reference_time = None
        self.reference_concentration = None

    def get_plot_data(
            self,
            model,
            initial_concentration,
            flux,
            tau_final
            ):
        """Evaluate ``model`` on the requested structured ``(rho, tau)`` grid.

        Returns radius, dimensionless time, physical time, and concentration.
        """
        device = next(model.parameters()).device

        self.model = model
        self.initial_concentration = float(initial_concentration)
        self.flux = float(flux)
        self.radius = np.linspace(
            0.0,
            1.0,
            self.number_rho_points
            )
        self.tau = np.linspace(
            0.0,
            tau_final,
            self.number_tau_points
            )

        rho_grid, tau_grid = np.meshgrid(
            self.radius,
            self.tau,
            indexing="ij"
            )
        model_inputs = np.column_stack(
            (
                rho_grid.ravel(),
                tau_grid.ravel()
            )
            )
        model_inputs = torch.tensor(
            model_inputs,
            dtype=torch.float32,
            device=device
            )

        model.eval()

        with torch.inference_mode():
            concentrations = model(
                model_inputs,
                initial_concentration
                )

        self.concentration = (
            concentrations
            .reshape(
                self.number_rho_points,
                self.number_tau_points
                )
            .detach()
            .cpu()
            .numpy()
            )
        self.time = self.tau * self.diffusion_time

        return self.radius, self.tau, self.time, self.concentration

    def plot_radial_profiles(self, n_profiles=5, show=True):
        self._require_plot_data()

        indices = np.linspace(
            0,
            len(self.time) - 1,
            n_profiles,
            dtype=int
            )

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
        axis.set_title("PINN radial concentration profiles")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_figure(fig, "radial_profile.png")
        self._finish_figure(fig, show)

        return fig, axis

    def plot_concentration_map(self, show=True):
        self._require_plot_data()

        fig, axis = plt.subplots(figsize=(9, 5))
        mesh = axis.pcolormesh(
            self.time,
            self.radius,
            self.concentration,
            shading="auto"
            )

        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel(r"Dimensionless radius, $\rho$")
        axis.set_title("PINN concentration field")
        fig.colorbar(
            mesh,
            ax=axis,
            label="Dimensionless concentration"
            )
        fig.tight_layout()

        self._save_figure(fig, "concentration_map.png")
        self._finish_figure(fig, show)

        return fig, axis

    def plot_boundary_concentrations(self, show=True):
        self._require_plot_data()

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
        axis.set_title("PINN boundary concentrations")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_figure(fig, "boundary_concentration.png")
        self._finish_figure(fig, show)

        return fig, axis

    def compute_average_concentration(self):
        self._require_plot_data()

        numerator = np.trapezoid(
            self.concentration * self.radius[:, None]**2,
            self.radius,
            axis=0
            )
        denominator = np.trapezoid(
            self.radius**2,
            self.radius
            )

        return numerator / denominator

    def compute_expected_mass_balance(self):
        self._require_plot_data()

        flux_profile = np.full_like(
            self.tau,
            self.flux,
            dtype=float
            )
        integrated_flux = cumulative_trapezoid(
            flux_profile,
            self.tau,
            initial=0.0
            )

        return (
            self.initial_concentration
            - 3.0 * integrated_flux
            )

    def plot_average_concentration(self, show=True):
        average_concentration = self.compute_average_concentration()

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            self.time,
            average_concentration,
            linewidth=2
            )

        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel("Average dimensionless concentration")
        axis.set_title("PINN volume-averaged concentration")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_figure(fig, "average_concentration.png")
        self._finish_figure(fig, show)

        return fig, axis

    def compute_mass_balance_residual(self):
        self._require_plot_data()

        average_concentration = self.compute_average_concentration()
        expected_concentration = self.compute_expected_mass_balance()

        return average_concentration - expected_concentration

    def verify_mass_balance(self, tolerance=1e-4):
        residual = self.compute_mass_balance_residual()
        passed = np.max(np.abs(residual)) < tolerance

        return passed, residual

    def verify_physical_bounds(self):
        self._require_plot_data()

        return np.all(
            (self.concentration >= 0.0)
            &
            (self.concentration <= 1.0)
            )

    def check_hard_constraints(self):
        """Return maximum initial-condition and centre-symmetry errors."""
        self._require_plot_data()

        initial_condition_max_error = float(
            np.max(
                np.abs(
                    self.concentration[:, 0]
                    - self.initial_concentration
                    )
                )
            )

        device = next(self.model.parameters()).device
        center_inputs = torch.column_stack(
            (
                torch.zeros(
                    len(self.tau),
                    dtype=torch.float32,
                    device=device
                    ),
                torch.tensor(
                    self.tau,
                    dtype=torch.float32,
                    device=device
                    )
            )
            ).requires_grad_(True)
        center_concentration = self.model(
            center_inputs,
            self.initial_concentration
            )
        center_derivatives = torch.autograd.grad(
            outputs=center_concentration,
            inputs=center_inputs,
            grad_outputs=torch.ones_like(center_concentration)
            )[0]
        center_symmetry_max_error = float(
            torch.max(
                torch.abs(center_derivatives[:, 0])
                ).detach().cpu()
            )

        return {
            "initial_condition_max_error": initial_condition_max_error,
            "center_symmetry_max_error": center_symmetry_max_error
            }

    def plot_mass_balance_residual(self, show=True):
        residual = self.compute_mass_balance_residual()

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            self.time,
            residual,
            linewidth=2
            )
        axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1
            )

        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel("Mass balance residual")
        axis.set_title("PINN mass balance residual")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_figure(fig, "mass_balance_residual.png")
        self._finish_figure(fig, show)

        return fig, axis

    def validate_surface_concentration(
            self,
            number_eigenfunctions=5,
            show=True
            ):
        """Compare the PINN surface history with the Guo--White approximation.

        Returns the figure, axis, and surface-concentration RMSE.
        """
        self._require_plot_data()

        surface_concentration = self.concentration[-1, :]
        tau_reference = np.linspace(
            0.0,
            min(float(self.tau[-1]), 0.68),
            300
            )
        reference_concentration = self._guo_white_surface_concentration(
            tau=tau_reference,
            initial_concentration=self.initial_concentration,
            flux=self.flux,
            number_eigenfunctions=number_eigenfunctions
            )
        pinn_surface_interpolated = np.interp(
            tau_reference,
            self.tau,
            surface_concentration
            )
        rmse = np.sqrt(
            np.mean(
                (reference_concentration - pinn_surface_interpolated)**2
                )
            )

        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            self.tau,
            surface_concentration,
            color="tab:blue",
            linewidth=2.5,
            label="PINN solution",
            zorder=2
            )
        axis.scatter(
            tau_reference,
            reference_concentration,
            color="tab:orange",
            marker="x",
            s=28,
            linewidths=1.2,
            label="Guo and White reference",
            zorder=3
            )

        axis.set_xlabel(r"Dimensionless time, $\tau$")
        axis.set_ylabel(r"Surface concentration, $C_s$")
        axis.set_title(
            rf"Surface concentration for $\delta={self.flux:.4f}$"
            )
        axis.text(
            0.98,
            0.05,
            f"RMSE = {rmse:.2e}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "gray"
                }
            )
        axis.set_xlim(left=0.0)
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_figure(fig, "surface_concentration_validation.png")
        self._finish_figure(fig, show)

        return fig, axis, rmse

    def animate_radial_profile(self, interval=40):
        self._require_plot_data()

        fig, axis = plt.subplots(figsize=(8, 5))
        line, = axis.plot([], [], linewidth=2)

        concentration_minimum = float(np.min(self.concentration))
        concentration_maximum = float(np.max(self.concentration))
        concentration_range = concentration_maximum - concentration_minimum
        padding = max(0.05 * concentration_range, 1e-6)

        axis.set_xlim(
            float(self.radius.min()),
            float(self.radius.max())
            )
        axis.set_ylim(
            concentration_minimum - padding,
            concentration_maximum + padding
            )
        axis.set_xlabel(r"Dimensionless radius, $\rho$")
        axis.set_ylabel("Dimensionless concentration")
        axis.grid(True, alpha=0.3)
        title = axis.set_title("")

        def initialize_animation():
            line.set_data([], [])
            return line, title

        def update_animation(frame):
            line.set_data(
                self.radius,
                self.concentration[:, frame]
                )
            title.set_text(
                f"PINN radial profile at t = {self.time[frame]:.0f} s"
                )
            return line, title

        animation = FuncAnimation(
            fig,
            update_animation,
            init_func=initialize_animation,
            frames=len(self.time),
            interval=interval,
            blit=True
            )

        plt.show()

        return animation

    def show_all(self, animate=True, show=True):
        """Generate all standalone PINN figures and optionally animate them."""
        self._require_plot_data()

        self.plot_radial_profiles(show=show)
        self.plot_concentration_map(show=show)
        self.plot_boundary_concentrations(show=show)
        self.plot_average_concentration(show=show)
        self.plot_mass_balance_residual(show=show)
        self.validate_surface_concentration(show=show)

        if animate and show:
            return self.animate_radial_profile()

        return None

    def load_reference_data(self, reference_dataset_path):
        """Load and align an FVM concentration field with the plotting grid.

        Returns aligned radius, dimensionless time, physical time, and field.
        """
        self._require_plot_data()

        reference_frame = pd.read_csv(reference_dataset_path)
        required_columns = {"rho", "tau", "concentration"}

        if not required_columns.issubset(reference_frame.columns):
            raise ValueError(
                "The reference dataset must contain rho, tau and "
                "concentration columns."
                )

        native_radius = np.sort(
            reference_frame["rho"].unique()
            )
        native_tau = np.sort(
            reference_frame["tau"].unique()
            )
        native_concentration = (
            reference_frame
            .pivot(
                index="rho",
                columns="tau",
                values="concentration"
                )
            .reindex(
                index=native_radius,
                columns=native_tau
                )
            .to_numpy()
            )

        same_radius_grid = (
            native_radius.shape == self.radius.shape
            and np.allclose(native_radius, self.radius)
            )
        same_tau_grid = (
            native_tau.shape == self.tau.shape
            and np.allclose(native_tau, self.tau)
            )

        if same_radius_grid and same_tau_grid:
            reference_concentration = native_concentration
        else:
            interpolator = RegularGridInterpolator(
                (native_radius, native_tau),
                native_concentration,
                bounds_error=True
                )
            rho_grid, tau_grid = np.meshgrid(
                self.radius,
                self.tau,
                indexing="ij"
                )
            interpolation_points = np.column_stack(
                (
                    rho_grid.ravel(),
                    tau_grid.ravel()
                )
                )
            reference_concentration = interpolator(
                interpolation_points
                ).reshape(
                    self.number_rho_points,
                    self.number_tau_points
                    )

        self.reference_radius = self.radius.copy()
        self.reference_tau = self.tau.copy()
        self.reference_time = self.time.copy()
        self.reference_concentration = reference_concentration

        return (
            self.reference_radius,
            self.reference_tau,
            self.reference_time,
            self.reference_concentration
            )

    def compute_reference_average_concentration(self):
        self._require_reference_data()

        numerator = np.trapezoid(
            self.reference_concentration
            * self.reference_radius[:, None]**2,
            self.reference_radius,
            axis=0
            )
        denominator = np.trapezoid(
            self.reference_radius**2,
            self.reference_radius
            )

        return numerator / denominator

    def compute_reference_flux_profile(self):
        self._require_reference_data()

        current_profile = np.zeros_like(
            self.reference_time,
            dtype=float
            )
        ramp_up_mask = (
            (self.reference_time >= 0.0)
            &
            (self.reference_time < self.ramp_time)
            )
        constant_current_mask = (
            (self.reference_time >= self.ramp_time)
            &
            (self.reference_time < self.ramp_down_start)
            )
        ramp_down_mask = (
            (self.reference_time >= self.ramp_down_start)
            &
            (
                self.reference_time
                < self.ramp_down_start + self.ramp_time
            )
            )

        current_profile[ramp_up_mask] = 0.5 * (
            1.0
            - np.cos(
                np.pi
                * self.reference_time[ramp_up_mask]
                / self.ramp_time
                )
            )
        current_profile[constant_current_mask] = 1.0
        current_profile[ramp_down_mask] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (
                    self.reference_time[ramp_down_mask]
                    - self.ramp_down_start
                )
                / self.ramp_time
                )
            )

        return self.flux * current_profile

    def compute_reference_expected_mass_balance(self):
        reference_flux = self.compute_reference_flux_profile()
        integrated_flux = cumulative_trapezoid(
            reference_flux,
            self.reference_tau,
            initial=0.0
            )

        return (
            self.initial_concentration
            - 3.0 * integrated_flux
            )

    def compute_reference_mass_balance_residual(self):
        reference_average = (
            self.compute_reference_average_concentration()
            )
        reference_expected = (
            self.compute_reference_expected_mass_balance()
            )

        return reference_average - reference_expected

    def verify_reference_mass_balance(self, tolerance=1e-4):
        residual = self.compute_reference_mass_balance_residual()
        passed = np.max(np.abs(residual)) < tolerance

        return passed, residual

    def verify_reference_physical_bounds(self):
        self._require_reference_data()

        return np.all(
            (self.reference_concentration >= 0.0)
            &
            (self.reference_concentration <= 1.0)
            )

    def compute_comparison_metrics(self):
        """Return global, boundary, average, and phase-wise PINN errors."""
        self._require_reference_data()

        pinn_average = self.compute_average_concentration()
        reference_average = (
            self.compute_reference_average_concentration()
            )

        metrics = {
            "field": self._error_metrics(
                self.reference_concentration,
                self.concentration
                ),
            "centre": self._error_metrics(
                self.reference_concentration[0, :],
                self.concentration[0, :]
                ),
            "surface": self._error_metrics(
                self.reference_concentration[-1, :],
                self.concentration[-1, :]
                ),
            "average": self._error_metrics(
                reference_average,
                pinn_average
                ),
            "phases": {}
            }

        phase_intervals = {
            "ramp_up": (0.0, self.ramp_time),
            "constant_current": (
                self.ramp_time,
                self.ramp_down_start
                ),
            "ramp_down": (
                self.ramp_down_start,
                self.ramp_down_start + self.ramp_time
                ),
            "rest": (
                self.ramp_down_start + self.ramp_time,
                float(self.time[-1])
                )
            }

        for phase_name, (start_time, end_time) in phase_intervals.items():
            phase_mask = (
                (self.time >= start_time)
                &
                (self.time <= end_time)
                )

            if np.any(phase_mask):
                metrics["phases"][phase_name] = self._error_metrics(
                    self.reference_concentration[:, phase_mask],
                    self.concentration[:, phase_mask]
                    )

        return metrics

    def run_comparative_checks(self, mass_balance_tolerance=1e-4):
        """Return mass, bounds, and hard-constraint checks for PINN and FVM."""
        self._require_reference_data()

        pinn_mass_passed, pinn_mass_residual = (
            self.verify_mass_balance(mass_balance_tolerance)
            )
        reference_mass_passed, reference_mass_residual = (
            self.verify_reference_mass_balance(
                mass_balance_tolerance
                )
            )
        hard_constraints = self.check_hard_constraints()

        return {
            "pinn_mass_balance_passed": bool(pinn_mass_passed),
            "pinn_mass_balance_max_abs": float(
                np.max(np.abs(pinn_mass_residual))
                ),
            "reference_mass_balance_passed": bool(
                reference_mass_passed
                ),
            "reference_mass_balance_max_abs": float(
                np.max(np.abs(reference_mass_residual))
                ),
            "pinn_physical_bounds_passed": bool(
                self.verify_physical_bounds()
                ),
            "reference_physical_bounds_passed": bool(
                self.verify_reference_physical_bounds()
                ),
            "pinn_minimum_concentration": float(
                np.min(self.concentration)
                ),
            "pinn_maximum_concentration": float(
                np.max(self.concentration)
                ),
            "reference_minimum_concentration": float(
                np.min(self.reference_concentration)
                ),
            "reference_maximum_concentration": float(
                np.max(self.reference_concentration)
                ),
            **hard_constraints
            }

    def plot_comparison_radial_profiles(
            self,
            n_profiles=5,
            show=True
            ):
        self._require_reference_data()

        indices = np.linspace(
            0,
            len(self.time) - 1,
            n_profiles,
            dtype=int
            )
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
                [0],
                [0],
                color="black",
                linewidth=2,
                linestyle="-",
                label="FVM"
                ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2,
                linestyle="--",
                label="PINN"
                )
            ]
        time_legend = axis.legend(
            handles=time_handles,
            title="Time",
            loc="upper left"
            )
        axis.add_artist(time_legend)
        axis.legend(
            handles=model_handles,
            loc="lower right"
            )

        axis.set_xlabel(r"Dimensionless radius, $\rho$")
        axis.set_ylabel(r"Dimensionless concentration, $C$")
        axis.set_title("FVM–PINN radial concentration comparison")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(*self._shared_concentration_limits())
        fig.tight_layout()

        self._save_comparison_figure(
            fig,
            "radial_profiles_comparison.png"
            )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_comparison_average_concentration(self, show=True):
        self._require_reference_data()

        reference_average = (
            self.compute_reference_average_concentration()
            )
        pinn_average = self.compute_average_concentration()
        fig, axis = plt.subplots(figsize=(8, 5))

        axis.plot(
            self.time,
            reference_average,
            linewidth=2,
            label="FVM"
            )
        axis.plot(
            self.time,
            pinn_average,
            linewidth=2,
            linestyle="--",
            label="PINN"
            )
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel("Average dimensionless concentration")
        axis.set_title("Volume-averaged concentration comparison")
        axis.set_ylim(*self._shared_concentration_limits())
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        self._save_comparison_figure(
            fig,
            "average_concentration_comparison.png"
            )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_comparison_boundary_concentrations(self, show=True):
        self._require_reference_data()

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharex=True,
            sharey=True
            )
        boundary_data = (
            (
                "Particle centre",
                self.reference_concentration[0, :],
                self.concentration[0, :]
            ),
            (
                "Particle surface",
                self.reference_concentration[-1, :],
                self.concentration[-1, :]
            )
            )

        for axis, (
                boundary_name,
                reference_values,
                pinn_values
                ) in zip(axes, boundary_data):
            axis.plot(
                self.time,
                reference_values,
                linewidth=2,
                label="FVM"
                )
            axis.plot(
                self.time,
                pinn_values,
                linewidth=2,
                linestyle="--",
                label="PINN"
                )
            axis.set_title(boundary_name)
            axis.set_xlabel("Physical time [s]")
            axis.grid(True, alpha=0.3)
            axis.legend()

        axes[0].set_ylabel("Dimensionless concentration")
        axes[0].set_ylim(*self._shared_concentration_limits())
        fig.suptitle("FVM–PINN boundary concentration comparison")
        fig.tight_layout()

        self._save_comparison_figure(
            fig,
            "boundary_concentrations_comparison.png"
            )
        self._finish_figure(fig, show)

        return fig, axes

    def plot_comparison_concentration_maps(self, show=True):
        self._require_reference_data()

        concentration_minimum = float(
            min(
                np.min(self.reference_concentration),
                np.min(self.concentration)
                )
            )
        concentration_maximum = float(
            max(
                np.max(self.reference_concentration),
                np.max(self.concentration)
                )
            )
        concentration_error = (
            self.reference_concentration
            - self.concentration
            )
        error_limit = float(
            np.max(np.abs(concentration_error))
            )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            sharex=True,
            sharey=True,
            constrained_layout=True
            )
        reference_mesh = axes[0].pcolormesh(
            self.time,
            self.radius,
            self.reference_concentration,
            shading="auto",
            vmin=concentration_minimum,
            vmax=concentration_maximum
            )
        axes[1].pcolormesh(
            self.time,
            self.radius,
            self.concentration,
            shading="auto",
            vmin=concentration_minimum,
            vmax=concentration_maximum
            )
        error_mesh = axes[2].pcolormesh(
            self.time,
            self.radius,
            concentration_error,
            shading="auto",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit
            )

        axes[0].set_title("FVM")
        axes[1].set_title("PINN")
        axes[2].set_title(r"Residual $C_{FVM}-C_{PINN}$")

        for axis in axes:
            axis.set_xlabel("Physical time [s]")
        axes[0].set_ylabel(r"Dimensionless radius, $\rho$")

        fig.colorbar(
            reference_mesh,
            ax=axes[:2],
            label="Dimensionless concentration"
            )
        fig.colorbar(
            error_mesh,
            ax=axes[2],
            label="Concentration residual"
            )
        fig.suptitle("Concentration field comparison")

        self._save_comparison_figure(
            fig,
            "concentration_fields_comparison.png"
            )
        self._finish_figure(fig, show)

        return fig, axes

    def plot_surface_concentration_residual(self, show=True):
        self._require_reference_data()

        surface_residual = (
            self.reference_concentration[-1, :]
            - self.concentration[-1, :]
            )
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            self.time,
            surface_residual,
            linewidth=2
            )
        axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1
            )
        axis.set_xlabel("Physical time [s]")
        axis.set_ylabel(r"$C_{s,FVM}-C_{s,PINN}$")
        axis.set_title("Surface concentration residual")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_comparison_figure(
            fig,
            "surface_concentration_residual.png"
            )
        self._finish_figure(fig, show)

        return fig, axis

    def plot_comparison_mass_balance_residuals(self, show=True):
        self._require_reference_data()

        reference_residual = (
            self.compute_reference_mass_balance_residual()
            )
        pinn_residual = self.compute_mass_balance_residual()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            sharex=True
            )

        residual_data = (
            (
                "FVM mass balance residual",
                reference_residual
            ),
            (
                "PINN mass balance residual",
                pinn_residual
            )
            )

        for axis, (title, residual) in zip(axes, residual_data):
            axis.plot(
                self.time,
                residual,
                linewidth=2
                )
            axis.axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1
                )
            axis.set_title(title)
            axis.set_xlabel("Physical time [s]")
            axis.set_ylabel(
                r"$\overline{C}-C_{balance}$"
                )
            axis.grid(True, alpha=0.3)
            axis.text(
                0.98,
                0.05,
                f"max |R| = {np.max(np.abs(residual)):.2e}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "gray"
                    }
                )

        fig.suptitle("Mass balance comparison using cumulative residuals")
        fig.tight_layout()

        self._save_comparison_figure(
            fig,
            "mass_balance_comparison.png"
            )
        self._finish_figure(fig, show)

        return fig, axes

    def show_comparison(self, show=True):
        """Generate the complete set of FVM--PINN comparison figures."""
        self._require_reference_data()

        self.plot_comparison_radial_profiles(show=show)
        self.plot_comparison_average_concentration(show=show)
        self.plot_comparison_boundary_concentrations(show=show)
        self.plot_comparison_concentration_maps(show=show)
        self.plot_surface_concentration_residual(show=show)
        self.plot_comparison_mass_balance_residuals(show=show)

    def _guo_white_surface_concentration(
            self,
            tau,
            initial_concentration,
            flux,
            number_eigenfunctions
            ):
        tau = np.asarray(tau, dtype=float)
        eigenvalues = []

        for eigenvalue_index in range(1, number_eigenfunctions + 2):
            lower_bound = eigenvalue_index * np.pi + 1e-10
            upper_bound = (
                eigenvalue_index * np.pi
                + np.pi / 2.0
                - 1e-10
                )
            root = brentq(
                lambda value: value - np.tan(value),
                lower_bound,
                upper_bound
                )
            eigenvalues.append(root)

        eigenvalues = np.asarray(eigenvalues)
        retained_eigenvalues = eigenvalues[:number_eigenfunctions]
        next_eigenvalue = eigenvalues[number_eigenfunctions]

        retained_inverse_square_sum = np.sum(
            1.0 / retained_eigenvalues**2
            )
        truncation_correction = (
            (
                1.0 / 10.0
                - retained_inverse_square_sum
            )
            * (
                1.0
                - np.exp(-next_eigenvalue**2 * tau)
            )
            + np.sqrt(tau / np.pi)
            * erfc(next_eigenvalue * np.sqrt(tau))
            )
        modal_terms = np.sum(
            (
                1.0 / retained_eigenvalues[:, None]**2
            )
            * (
                1.0
                - np.exp(
                    -retained_eigenvalues[:, None]**2
                    * tau[None, :]
                    )
            ),
            axis=0
            )

        return (
            initial_concentration
            - 3.0 * flux * tau
            - 2.0 * flux * truncation_correction
            - 2.0 * flux * modal_terms
            )

    def _require_plot_data(self):
        if any(
            value is None
            for value in (
                self.radius,
                self.tau,
                self.time,
                self.concentration
            )
        ):
            raise RuntimeError(
                "Call get_plot_data() before generating plots."
                )

    def _require_reference_data(self):
        self._require_plot_data()

        if any(
            value is None
            for value in (
                self.reference_radius,
                self.reference_tau,
                self.reference_time,
                self.reference_concentration
            )
        ):
            raise RuntimeError(
                "Call load_reference_data() before running FVM–PINN "
                "comparisons."
                )

    def _shared_concentration_limits(self):
        self._require_reference_data()

        concentration_minimum = float(
            min(
                np.min(self.reference_concentration),
                np.min(self.concentration)
                )
            )
        concentration_maximum = float(
            max(
                np.max(self.reference_concentration),
                np.max(self.concentration)
                )
            )
        concentration_range = (
            concentration_maximum - concentration_minimum
            )
        padding = max(
            0.05 * concentration_range,
            1e-6
            )

        return (
            concentration_minimum - padding,
            concentration_maximum + padding
            )

    @staticmethod
    def _error_metrics(reference, prediction):
        error = prediction - reference
        reference_norm = np.linalg.norm(reference)

        if reference_norm == 0.0:
            raise ValueError(
                "Cannot compute a relative error with a zero reference norm."
                )

        return {
            "relative_l2": float(
                np.linalg.norm(error) / reference_norm
                ),
            "rmse": float(
                np.sqrt(np.mean(error**2))
                ),
            "mae": float(
                np.mean(np.abs(error))
                ),
            "maximum_absolute_error": float(
                np.max(np.abs(error))
                )
            }

    def _save_figure(self, figure, filename):
        if self.save:
            figure.savefig(
                os.path.join(self.images_folder, filename),
                dpi=300,
                bbox_inches="tight"
                )

    def _save_comparison_figure(self, figure, filename):
        if self.save:
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
