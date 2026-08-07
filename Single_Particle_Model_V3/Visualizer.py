import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
from scipy.optimize import brentq
from scipy.special import erfc


class SimulationVisualizer(object):
    """Visualize and validate finite-volume particle simulations.

    It generates field, profile, boundary, mass, and temperature figures and
    can compare the surface solution with the Guo--White approximation.
    """

    def __init__(
            self,
            tau,
            time,
            radius,
            concentration,
            save_im = True
            ):

        self.tau = np.asarray(tau)
        self.time = np.asarray(time)
        self.radius = np.asarray(radius)
        self.concentration = np.asarray(concentration)

        self.save = save_im
        self.images_folder = './Images'
        os.makedirs(self.images_folder,exist_ok=True)

    def plot_radial_profiles(
            self,
            n_profiles=5
            ):

        """
        Plot radial concentration profiles at different times.
        """

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
                self.concentration[:, idx],
                linewidth=2,
                label=f"t = {self.time[idx]:.0f} s"
            )

        plt.xlabel("Dimensionless radius, $\\rho$")
        plt.ylabel("Dimensionless concentration, $C$")
        plt.title("Radial concentration profiles")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if self.save:
            plt.savefig(self.images_folder + '/radial_profile.png')

        plt.show()

    def plot_concentration_map(self):

        """
        Plot the complete concentration field.
        """

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

        if self.save:
            plt.savefig(self.images_folder + '/concentration_map.png')

        plt.show()

    def plot_boundary_concentrations(self):

        """
        Plot centre and surface concentration histories.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.time,
            self.concentration[0, :],
            linewidth=2,
            label="Particle centre"
        )

        plt.plot(
            self.time,
            self.concentration[-1, :],
            linewidth=2,
            label="Particle surface"
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Dimensionless concentration")
        plt.title("Boundary concentrations")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if self.save:
            plt.savefig(self.images_folder + '/boundary_concentration.png')

        plt.show()

    def plot_average_concentration(
            self,
            average_concentration
            ):

        """
        Plot the volume-averaged concentration.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.time,
            average_concentration,
            linewidth=2
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Average concentration")
        plt.title("Volume-averaged concentration")

        plt.grid(True)
        plt.tight_layout()

        if self.save:
            plt.savefig(self.images_folder + '/average_concentration.png')

        plt.show()

    def plot_mass_balance_residual(
        self,
        residual
        ):
        """
        Plot the mass balance residual.
        """

        plt.figure(figsize=(8,5))

        plt.plot(
            self.time,
            residual,
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

        if self.save:
            plt.savefig(self.images_folder + '/mass_balance_residual.png')
            
        plt.show()

    def animate_radial_profile(
        self,
        interval=40
        ):

        """
        Animate the radial concentration profile.
        """

        fig, ax = plt.subplots(figsize=(8,5))

        line, = ax.plot(
            [],
            [],
            linewidth=2
        )

        ax.set_xlim(
            self.radius.min(),
            self.radius.max()
        )

        ax.set_ylim(
            np.min(self.concentration),
            np.max(self.concentration)
        )

        ax.set_xlabel("Dimensionless radius, $\\rho$")
        ax.set_ylabel("Dimensionless concentration")
        ax.grid(True)

        title = ax.set_title("")

        def init():
            line.set_data([], [])

            return line,

        def update(frame):
            line.set_data(
                self.radius,
                self.concentration[:,frame]
            )
            title.set_text(
                f"t = {self.time[frame]:.0f} s"
            )

            return line,

        animation = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(self.time),
            interval=interval,
            blit=True
        )

        plt.show()

        return animation

    def validate_surface_concentration(self,concentration,tau,profile):
        surface_concentration = concentration[-1, :]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            tau,
            surface_concentration,
            color="tab:blue",
            linewidth=2.5,
            label="FVM solution",
            zorder=2
        )

        def guo_white_surface_concentration(
                tau,
                C0=0.5,
                delta=-0.2,
                N_eigenfunctions=5
                ):
            """
            Guo and White approximate surface-concentration solution
            for a constant dimensionless surface flux.

            Based on Eq. (46) of Guo and White (2012).
            """

            tau = np.asarray(tau, dtype=float)

            # Eigenvalues satisfying:
            # lambda_n - tan(lambda_n) = 0
            eigenvalues = []

            for n in range(1, N_eigenfunctions + 2):

                lower_bound = n * np.pi + 1e-10
                upper_bound = n * np.pi + np.pi / 2.0 - 1e-10

                root = brentq(
                    lambda value: value - np.tan(value),
                    lower_bound,
                    upper_bound
                )

                eigenvalues.append(root)

            eigenvalues = np.asarray(eigenvalues)

            lambda_retained = eigenvalues[:N_eigenfunctions]
            lambda_next = eigenvalues[N_eigenfunctions]

            retained_inverse_square_sum = np.sum(
                1.0 / lambda_retained**2
            )

            truncation_correction = (
                (
                    1.0 / 10.0
                    - retained_inverse_square_sum
                )
                * (
                    1.0
                    - np.exp(-lambda_next**2 * tau)
                )
                + np.sqrt(tau / np.pi)
                * erfc(lambda_next * np.sqrt(tau))
            )

            modal_terms = np.sum(
                (
                    1.0 / lambda_retained[:, None]**2
                )
                * (
                    1.0
                    - np.exp(
                        -lambda_retained[:, None]**2
                        * tau[None, :]
                    )
                ),
                axis=0
            )

            surface_concentration = (
                C0
                - 3.0 * delta * tau
                - 2.0 * delta * truncation_correction
                - 2.0 * delta * modal_terms
            )

            return surface_concentration

        if profile == 'constant':
            tau_reference = np.linspace(
                0.0,
                0.68,
                300
            )

            Cs_reference = guo_white_surface_concentration(
                tau=tau_reference,
                C0=0.5,
                delta=-0.2,
                N_eigenfunctions=5
            )

            if tau_reference.size > 0 and Cs_reference.size > 0:
                if tau_reference.shape != Cs_reference.shape:
                    raise ValueError(
                        "tau_reference and Cs_reference must have the same shape"
                    )

                ax.scatter(
                    tau_reference,
                    Cs_reference,
                    color="tab:orange",
                    marker="x",
                    s=28,
                    linewidths=1.2,
                    label="Guo and White reference",
                    zorder=3
                )

                surface_concentration_interp = np.interp(tau_reference,tau,surface_concentration)
                rmse = np.sqrt(np.mean((Cs_reference - surface_concentration_interp)**2))

        ax.set_xlabel(r"Dimensionless time, $\tau$")
        ax.set_ylabel(r"Surface concentration, $C_s$")
        if profile == 'constant':
            ax.set_title(r"Surface concentration for $\delta(\tau)=-0.2$")
            ax.text(
                0.98,
                0.05,
                f"RMSE = {rmse:.2e}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray"
                )
            )
        else:
            ax.set_title("Surface concentration under time-varying intercalation flux")
        ax.set_xlim(left=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        plt.show()

    def plot_temperature_map(self,temperatures):
        
        """
        Plot the complete temperature field.
        """

        plt.figure(figsize=(9, 5))


        temperature_field = np.tile(temperatures[None, :],
                                    (self.radius.size, 1))
        mesh = plt.pcolormesh(
            self.time,
            self.radius,
            temperature_field,
            shading="auto",
            vmin=temperatures.min(),
            vmax=temperatures.max()
        )

        plt.xlabel("Physical time [s]")
        plt.ylabel("Dimensionless radius, $\\rho$")
        plt.title("Temperature field")

        colorbar = plt.colorbar(mesh)
        colorbar.set_label("Temperature [K]")

        colorbar.formatter.set_useOffset(False)
        colorbar.formatter.set_scientific(False)
        colorbar.update_ticks()

        plt.tight_layout()

        if self.save:
            plt.savefig(self.images_folder + '/temperature_map.png')

        plt.show()

        plt.figure(figsize=(10,5))
        plt.plot(self.time, temperatures)
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [K]")

        axis = plt.gca()
        axis.ticklabel_format(
            axis="y",
            style="plain",
            useOffset=False)
        
        plt.title("Temperature Variation in Time")
        plt.grid(True)

        if self.save:
            plt.savefig(self.images_folder + '/temperature_curve.png')

        plt.show()

    def show_all(
            self,
            average_concentration,
            residual,
            concentration,
            tau,
            profile,
            temperatures
            ):

        """
        Display all available plots.
        """

        self.plot_radial_profiles()
        self.plot_concentration_map()
        self.plot_boundary_concentrations()
        self.plot_average_concentration(average_concentration)
        self.plot_mass_balance_residual(residual)
        self.validate_surface_concentration(concentration,tau,profile)
        self.animate_radial_profile()
        self.plot_temperature_map(temperatures)
