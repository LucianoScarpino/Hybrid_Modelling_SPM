from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


class HybridSimulationVisualizer(object):
    """Visualize a corrected concentration field and its mass consistency.

    Corrected and reference datasets are aligned once, then reused to generate
    radial, boundary, average, field, and mass-balance figures.
    """

    def __init__(
            self,
            corrected_dataset,
            simplified_dataset,
            reference_dataset,
            inventory_offset,
            images_folder,
            compare=False,
            save_im=True,
            show_im=True
            ):

        corrected_dataset = self.read_dataset(corrected_dataset)
        simplified_dataset = self.read_dataset(simplified_dataset)
        reference_dataset = self.read_dataset(reference_dataset)

        corrected_profiles = (
            corrected_dataset
            .pivot(index="tau", columns="rho", values="concentration")
            .sort_index()
            .sort_index(axis=1)
        )

        corrected_states = (
            corrected_dataset
            .drop_duplicates("tau")
            .set_index("tau")
            .sort_index()
            .loc[corrected_profiles.index]
        )

        simplified_profiles = (
            simplified_dataset
            .pivot(index="tau", columns="rho", values="concentration")
            .sort_index()
            .sort_index(axis=1)
        )

        reference_profiles = (
            reference_dataset
            .pivot(index="tau", columns="rho", values="concentration")
            .sort_index()
            .sort_index(axis=1)
        )

        reference_states = (
            reference_dataset
            .drop_duplicates("tau")
            .set_index("tau")
            .sort_index()
            .loc[reference_profiles.index]
        )

        reference_tau = reference_profiles.index.to_numpy(dtype=float)
        simplified_tau = simplified_profiles.index.to_numpy(dtype=float)
        corrected_tau = corrected_profiles.index.to_numpy(dtype=float)

        if (
                not np.allclose(reference_tau,corrected_tau)
                or not np.allclose(reference_tau,simplified_tau)
                ):
            raise ValueError(
                "Reference, simplified, and corrected tau grids do not match."
            )

        reference_radius = reference_profiles.columns.to_numpy(dtype=float)
        simplified_radius = simplified_profiles.columns.to_numpy(dtype=float)
        corrected_radius = corrected_profiles.columns.to_numpy(dtype=float)

        if (
                not np.allclose(reference_radius,corrected_radius)
                or not np.allclose(reference_radius,simplified_radius)
                ):
            raise ValueError(
                "Reference, simplified, and corrected radial grids do not match."
            )

        self.inventory_offset = float(inventory_offset)
        self.compare = bool(compare)

        if not np.isfinite(self.inventory_offset):
            raise ValueError("The inventory offset must be finite.")

        self.tau = corrected_tau
        self.time = corrected_states["time"].to_numpy(dtype=float)
        self.radius = corrected_radius
        self.concentration = corrected_profiles.to_numpy(dtype=float).T
        self.average_weights = self.compute_average_weights(self.radius)
        self.average_concentration = np.sum(
            corrected_profiles.to_numpy(dtype=float)
            * self.average_weights[None,:],
            axis=1
        )
        self.simplified_average_concentration = np.sum(
            simplified_profiles.to_numpy(dtype=float)
            * self.average_weights[None,:],
            axis=1
        )
        self.reference_average_concentration = np.sum(
            reference_profiles.to_numpy(dtype=float)
            * self.average_weights[None,:],
            axis=1
        )
        self.average_correction = (
            self.average_concentration
            - self.simplified_average_concentration
        )

        inventory_constraint_error = np.max(
            np.abs(self.average_correction - self.inventory_offset)
        )

        if inventory_constraint_error > 1e-6:
            raise ValueError(
                "The corrected dataset and checkpoint inventory offset do not "
                "match. Run HybridDemo.py again."
            )

        self.dimensionless_flux = reference_states[
            "dimensionless_flux"
        ].to_numpy(dtype=float)

        self.mass_balance_residual = self.compute_mass_balance_residual()

        self.save = save_im
        self.show = show_im
        self.images_folder = Path(images_folder)
        self.images_folder.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def read_dataset(dataset):
        if isinstance(dataset,pd.DataFrame):
            return dataset.copy()

        return pd.read_csv(dataset)

    @staticmethod
    def compute_average_weights(radius):
        """Return normalized trapezoidal weights for a spherical average."""
        radial_steps = np.diff(radius)

        if np.any(radial_steps <= 0.0):
            raise ValueError("The radial grid must be strictly increasing.")

        trapezoidal_weights = np.empty_like(radius)
        trapezoidal_weights[0] = radial_steps[0] / 2.0
        trapezoidal_weights[-1] = radial_steps[-1] / 2.0
        trapezoidal_weights[1:-1] = (
            radial_steps[:-1] + radial_steps[1:]
        ) / 2.0
        spherical_weights = trapezoidal_weights * radius ** 2

        return spherical_weights / np.sum(spherical_weights)

    def compute_mass_balance_residual(self):
        expected_average_concentration = (
            self.average_concentration[0]
            - 3.0 * cumulative_trapezoid(
                self.dimensionless_flux,
                self.tau,
                initial=0.0
            )
        )

        return (
            self.average_concentration
            - expected_average_concentration
        )

    def finalize_plot(self,filename):
        if self.save:
            plt.savefig(self.images_folder / filename)

        if self.show:
            plt.show()
        else:
            plt.close()

    def plot_radial_profiles(
            self,
            n_profiles=5
            ):

        """Plot radial concentration profiles at different times."""

        indices = np.linspace(
            0,
            len(self.time) - 1,
            n_profiles,
            dtype=int
        )

        plt.figure(figsize=(8, 5))

        for idx in indices:
            plt.plot(
                self.radius,
                self.concentration[:,idx],
                linewidth=2,
                label=f"t = {self.time[idx]:.0f} s"
            )

        plt.xlabel("Dimensionless radius, $\\rho$")
        plt.ylabel("Dimensionless concentration, $C$")
        plt.title("Radial concentration profiles")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        self.finalize_plot("radial_profile.png")

    def plot_concentration_map(self):
        """Plot the complete concentration field."""

        plt.figure(figsize=(9, 5))

        mesh = plt.pcolormesh(
            self.time,
            self.radius,
            self.concentration,
            shading="auto"
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Dimensionless radius, $\\rho$")
        plt.title("Concentration field")

        plt.colorbar(
            mesh,
            label="Dimensionless concentration"
        )

        plt.tight_layout()

        self.finalize_plot("concentration_map.png")

    def plot_boundary_concentrations(self):
        """Plot centre and surface concentration histories."""

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.time,
            self.concentration[0,:],
            linewidth=2,
            label="Particle centre"
        )

        plt.plot(
            self.time,
            self.concentration[-1,:],
            linewidth=2,
            label="Particle surface"
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Dimensionless concentration")
        plt.title("Boundary concentrations")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        self.finalize_plot("boundary_concentration.png")

    def plot_average_concentration(self):
        """Plot the hybrid average, optionally with comparison curves."""

        plt.figure(figsize=(8, 5))

        if self.compare:
            plt.plot(
                self.time,
                self.reference_average_concentration,
                linewidth=2,
                label="Reference"
            )
            plt.plot(
                self.time,
                self.simplified_average_concentration,
                linewidth=2,
                label="Simplified"
            )

        plt.plot(
            self.time,
            self.average_concentration,
            linewidth=2,
            label="Hybrid"
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Average concentration")
        plt.title("Volume-averaged concentration")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        self.finalize_plot("average_concentration.png")

    def plot_inventory_offset(self):
        """Plot the average correction and its constant target value."""
        plt.figure(figsize=(8,5))

        plt.plot(
            self.time,
            self.average_correction,
            linewidth=2,
            label=r"$\overline{\Delta C}_{\phi}$"
        )
        plt.axhline(
            self.inventory_offset,
            color="tab:red",
            linestyle="--",
            label=r"$b$"
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Average correction")
        plt.title("Constant volume-average correction")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        self.finalize_plot("inventory_offset.png")

    def plot_mass_balance_residual(self):
        """Plot the mass balance residual."""

        plt.figure(figsize=(8,5))

        plt.plot(
            self.time,
            self.mass_balance_residual,
            linewidth=2
        )

        plt.axhline(
            0.0,
            linestyle="--"
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Residual")
        plt.title("Mass balance residual")

        plt.grid(True)
        plt.tight_layout()

        self.finalize_plot("mass_balance_residual.png")

    def show_all(self):
        """Display all plots available for the hybrid model."""

        self.plot_radial_profiles()
        self.plot_concentration_map()
        self.plot_boundary_concentrations()
        self.plot_average_concentration()
        self.plot_inventory_offset()
        self.plot_mass_balance_residual()
