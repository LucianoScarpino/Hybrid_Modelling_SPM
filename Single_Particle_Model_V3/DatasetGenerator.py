import numpy as np
import os

from scipy.integrate import cumulative_trapezoid

class PostProcessing(object):
    def __init__(
        self,
        tau,
        time,
        radius,
        concentration,
        currents,
        intercalation_fluxes,
        dimensionless_fluxes,
        initial_concentration,
        temperatures
    ):

        self.tau = tau
        self.time = time
        self.radius = radius
        self.concentration = concentration

        self.current = currents
        self.intercalation_flux = intercalation_fluxes
        self.dimensionless_flux = dimensionless_fluxes

        self.C0 = initial_concentration
        self.temperatures = temperatures

    def compute_expected_mass_balance(self):

        integral_flux = cumulative_trapezoid(
            self.dimensionless_flux,
            self.tau,
            initial=0.0
        )

        expected_average_concentration = (
            self.C0 - 3.0 * integral_flux
        )

        return expected_average_concentration

    def compute_average_concentration(self):
        numerator = np.trapezoid(
            self.concentration * self.radius[:, None] ** 2,
            self.radius,
            axis=0
        )

        denominator = np.trapezoid(
            self.radius ** 2,
            self.radius
        )

        return numerator / denominator

    def compute_mass_balance_residual(self):
        C_bar = self.compute_average_concentration()
        C_bal = self.compute_expected_mass_balance()
        residual = C_bar - C_bal

        return residual

    def verify_mass_balance(self, tolerance=1e-4):
        residual = self.compute_mass_balance_residual()
        passed = np.max(np.abs(residual)) < tolerance

        return passed, residual

    def verify_physical_bounds(self):

        return np.all(
            (self.concentration >= 0.0)
            &
            (self.concentration <= 1.0)
        )

    def verify_temperature(
        self,
        minimum_temperature=250.0,
        maximum_temperature=400.0
    ):
        """
        Verify that the temperature history is numerically valid
        and remains within physical bounds.
        """

        checks = {
            "contains_nan": np.isnan(self.temperatures).any(),
            "contains_inf": np.isinf(self.temperatures).any(),
            "positive_temperature": np.all(self.temperatures > 0.0),
            "within_bounds": np.all(
                (self.temperatures >= minimum_temperature)
                &
                (self.temperatures <= maximum_temperature)
            )
        }

        passed = (
            not checks["contains_nan"]
            and not checks["contains_inf"]
            and checks["positive_temperature"]
            and checks["within_bounds"]
        )

        return passed, checks

    def generate_dataset(self):
        """
        Generate the dataset.

        Each sample corresponds to one space-time pair (rho_i, tau_k):

            D_i,k = {
                rho,
                tau,
                time,
                current,
                intercalation_flux,
                dimensionless_flux,
                concentration,
                temperature
            }

        Returns
        -------
        dataset : dict
            Flattened dataset.

        global_outputs : dict
            Surface, average concentration and temperature histories.
        """

        n_radius = len(self.radius)
        n_time = len(self.tau)

        dataset = {
            "rho": np.tile(self.radius, n_time),
            "tau": np.repeat(self.tau, n_radius),
            "time": np.repeat(self.time, n_radius),
            "current": np.repeat(self.current, n_radius),
            "intercalation_flux": np.repeat(
                self.intercalation_flux,
                n_radius
            ),
            "dimensionless_flux": np.repeat(
                self.dimensionless_flux,
                n_radius
            ),

            # Shape (Nr, Nt) -> (Nr*Nt,)
            "concentration": self.concentration.flatten(order="F"),
            "temperature": np.repeat(self.temperatures,n_radius)
        }

        global_outputs = {
            "surface_concentration":
                self.concentration[-1, :],
            "average_concentration":
                self.compute_average_concentration(),
            "temperature":
                self.temperatures
        }

        return dataset, global_outputs

    def get_pinn_training_arrays(self):
        dataset, global_outputs = self.generate_dataset()

        X = np.column_stack((   dataset["rho"],
                                dataset["tau"],
                            ))

        y = dataset["concentration"].reshape(-1, 1)

        return X, y, global_outputs

    def save_dataset(self,
                    folder_path,
                    dataset,
                    X,
                    y,
                    full_dataset_filename="full_simulation_dataset.csv",
                    pinn_dataset_filename="pinn_training_dataset.csv",
                    ):

        os.makedirs(folder_path,exist_ok=True)

        full_dataset_path = os.path.join(folder_path,full_dataset_filename)
        pinn_dataset_path = os.path.join(folder_path,pinn_dataset_filename)
        full_dataset_matrix = np.column_stack((dataset["rho"],
                                               dataset["tau"],
                                               dataset["time"],
                                               dataset["current"],
                                               dataset["intercalation_flux"],
                                               dataset["dimensionless_flux"],
                                               dataset["concentration"],
                                               dataset["temperature"]
                                               ))

        np.savetxt(
            full_dataset_path,
            full_dataset_matrix,
            delimiter=",",
            header=("rho,tau,time,current,"
                    "intercalation_flux,"
                    "dimensionless_flux,"
                    "concentration,"
                    "temperature"),
            comments=""
            )

        pinn_dataset_matrix = np.column_stack((X,y))

        np.savetxt(
            pinn_dataset_path,
            pinn_dataset_matrix,
            delimiter=",",
            header="rho,tau,concentration",
            comments=""
            )

        return full_dataset_path, pinn_dataset_path

    def save_simplified_dataset(
        self,
        tau,
        time,
        radius,
        concentration,
        average_concentration,
        surface_concentration,
        current_profile,
        folder_path="./SPMDataset",
        simplified_dataset_name="simplified_simulation_dataset.csv"
        ):

        tau = np.asarray(tau)
        time = np.asarray(time)
        radius = np.asarray(radius)
        concentration = np.asarray(concentration)
        average_concentration = np.asarray(average_concentration)
        surface_concentration = np.asarray(surface_concentration)
        current_profile = np.asarray(current_profile)

        n_radius = len(radius)
        n_time = len(tau)

        if concentration.shape != (n_radius, n_time):
            raise ValueError(
                "The concentration field must have shape "
                f"({n_radius}, {n_time}), received {concentration.shape}."
            )

        temporal_arrays = [
            time,
            average_concentration,
            surface_concentration,
            current_profile,
        ]

        if any(array.shape != (n_time,) for array in temporal_arrays):
            raise ValueError(
                "All time-dependent quantities must have shape "
                f"({n_time},)."
            )

        dataset_matrix = np.column_stack((
            np.tile(radius, n_time),
            np.repeat(tau, n_radius),
            np.repeat(time, n_radius),
            concentration.flatten(order="F"),
            np.repeat(average_concentration, n_radius),
            np.repeat(surface_concentration, n_radius),
            np.repeat(current_profile, n_radius),
        ))

        os.makedirs(folder_path, exist_ok=True)
        save_in = os.path.join(folder_path, simplified_dataset_name)

        np.savetxt(
            save_in,
            dataset_matrix,
            delimiter=",",
            header=(
                "rho,tau,time,"
                "concentration,"
                "average_concentration,"
                "surface_concentration,"
                "current"
            ),
            comments=""
        )

        return save_in