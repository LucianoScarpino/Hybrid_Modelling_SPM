import numpy as np
import torch

from pathlib import Path
from datetime import datetime
from copy import deepcopy

from NeuralNetwork import FFN


class ResidualLearner(object):
    """Train and assess the post-training Hybrid-PINN correction.

    Besides fitting ``C_FVM-C_PINN``, the class evaluates bounds, exact hard
    constraints, mass balance, surface flux, PDE consistency, and regularity.
    """

    def __init__(self):
        pass

    def learn_residual(
            self,
            train_res_dataset,
            val_res_dataset,
            epochs,
            lr,
            early_stopping_patience,
            device
            ):
        """Fit the residual FFN with an MSE and concentration-bounds penalty.

        Inputs are training settings and residual loaders. Returns the best
        validation model selected by early stopping.
        """

        print("Training Residual Learner...")

        model = FFN().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)

        best_val_loss = float("inf")
        best_params = None
        patience = 0
    
        for epoch in range(epochs):
            model.train()
            train_batch_loss = 0.0
            batch_num_points = 0

            for X,y in train_res_dataset:
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                pred = model(X)

                batch_bound_term = self.check_boundary_limits(pred + X[:,2:3],verbose=False)
                train_error = torch.mean((pred - y).square()) + batch_bound_term

                train_error.backward()
                optimizer.step()

                train_batch_loss += train_error.item() * X.shape[0]
                batch_num_points += X.shape[0]


            train_loss = train_batch_loss/batch_num_points

            val_loss = 0.0
            num_val_points = 0

            model.eval()

            for X,y in val_res_dataset:
                X = X.to(device)
                y = y.to(device)

                with torch.no_grad():
                    pred = model(X)

                    val_batch_bound_term = self.check_boundary_limits(pred + X[:,2:3],verbose=False)
                    res = torch.mean((pred - y).square()) + val_batch_bound_term

                    val_loss += res.item() * X.shape[0]
                    num_val_points += X.shape[0]

            val_loss = val_loss/num_val_points

            print(f"epoch [{epoch}/{epochs}] |"
                  f"training loss: {train_loss:.6e} |"
                  f"validation loss: {val_loss:.6e} |"
                  )

            print("-"*100)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_params = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print(f"Early stopped a epoch: {epoch}")
                    break

        model.load_state_dict(best_params)
        self.save_resiudal_checkpoint(
            model,
            best_val_loss,
            optimizer_name="ADAM"
        )

        print("Training Completed...")
        print("-"*100)

        return model

    def save_resiudal_checkpoint(
            self,
            model,
            validation_loss,
            optimizer_name,
            out_folder="./Results/Models",
            filename="residual_model.pth"
            ):
        """Save residual-model weights and validation metadata; return the path."""

        output_folder = Path(out_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename_path = Path(filename)

        checkpoint_path = output_folder / (f"{filename_path.stem}_{timestamp}"
                                            f"{filename_path.suffix}")

        collision_index = 1
        while checkpoint_path.exists():
            checkpoint_path = output_folder / (
                                                f"{filename_path.stem}_{timestamp}_"
                                                f"{collision_index}{filename_path.suffix}"
                                                )
            collision_index += 1

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "validation_loss": float(validation_loss),
            "optimizer": optimizer_name,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Model checkpoint saved to: "
            f"{checkpoint_path.resolve()}"
        )

        return checkpoint_path

    def check_boundary_limits(
            self,
            concentrations,
            low_b=0.0,
            high_b=1.0,
            verbose=True
            ):
        """Return a squared soft penalty for concentrations outside ``[low_b, high_b]``."""

        if verbose:
            max_C = concentrations.max().item()
            min_C = concentrations.min().item()
            
            print(
                f"Max boundary condition satified: {max_C < high_b}"
                f"Min boundary condition satified: {min_C > low_b}"
            )

        lower_violation = torch.relu(low_b - concentrations)
        upper_violation = torch.relu(concentrations - high_b)

        bounds_loss = torch.mean(lower_violation**2 + upper_violation**2)


        return bounds_loss

    @staticmethod
    def synchronize_device(selected_device):
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        elif selected_device.type == "mps":
            torch.mps.synchronize()

    @staticmethod
    def error_metrics(reference, prediction):
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
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "maximum_absolute_error": float(np.max(np.abs(error)))
        }

    @staticmethod
    def volume_average(concentration, radius):
        numerator = np.trapezoid(
            concentration * radius[:, None]**2,
            radius,
            axis=0
        )
        denominator = np.trapezoid(radius**2, radius)

        return numerator / denominator

    @staticmethod
    def concentration_field(dataset, column, radius, tau):
        return (
            dataset
            .pivot(index="rho", columns="tau", values=column)
            .reindex(index=radius, columns=tau)
            .to_numpy()
        )

    def compute_comparison_metrics(
            self,
            dataset,
            prediction_column,
            ramp_time=300.0,
            ramp_down_start=3600.0
            ):
        """Compare one predicted field with FVM globally and by operating phase.

        Parameters are the corrected dataset and prediction column. Returns
        field, boundary, average, and phase-wise error dictionaries.
        """
        radius = np.sort(dataset["rho"].unique())
        tau = np.sort(dataset["tau"].unique())
        time = (
            dataset[["tau", "time"]]
            .drop_duplicates("tau")
            .sort_values("tau")["time"]
            .to_numpy()
        )
        reference = self.concentration_field(
            dataset,
            "concentration_reference",
            radius,
            tau
        )
        prediction = self.concentration_field(
            dataset,
            prediction_column,
            radius,
            tau
        )
        reference_average = self.volume_average(reference, radius)
        prediction_average = self.volume_average(prediction, radius)

        metrics = {
            "field": self.error_metrics(reference, prediction),
            "centre": self.error_metrics(
                reference[0, :],
                prediction[0, :]
            ),
            "surface": self.error_metrics(
                reference[-1, :],
                prediction[-1, :]
            ),
            "average": self.error_metrics(
                reference_average,
                prediction_average
            ),
            "phases": {}
        }
        phase_intervals = {
            "ramp_up": (0.0, ramp_time),
            "constant_current": (ramp_time, ramp_down_start),
            "ramp_down": (
                ramp_down_start,
                ramp_down_start + ramp_time
            ),
            "rest": (
                ramp_down_start + ramp_time,
                float(time[-1])
            )
        }

        for phase_name, (start_time, end_time) in phase_intervals.items():
            phase_mask = (time >= start_time) & (time <= end_time)

            if np.any(phase_mask):
                metrics["phases"][phase_name] = self.error_metrics(
                    reference[:, phase_mask],
                    prediction[:, phase_mask]
                )

        return metrics

    def compute_physical_checks(
            self,
            dataset,
            initial_concentration,
            mass_balance_tolerance=1e-4
            ):
        """Evaluate mass balance and concentration bounds for all three fields.

        Returns diagnostics for the FVM reference, PINN, and Hybrid-PINN.
        """
        radius = np.sort(dataset["rho"].unique())
        tau = np.sort(dataset["tau"].unique())
        flux_profile = (
            dataset[["tau", "dimensionless_flux"]]
            .drop_duplicates("tau")
            .sort_values("tau")
        )
        flux = flux_profile["dimensionless_flux"].to_numpy()
        integrated_flux = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(
                    0.5
                    * (flux[1:] + flux[:-1])
                    * np.diff(tau)
                )
            )
        )
        expected_average = initial_concentration - 3.0 * integrated_flux
        checks = {}

        for name, column in (
                ("reference", "concentration_reference"),
                ("pinn", "concentration_pinn"),
                ("hybrid", "concentration_hybrid")
                ):
            field = self.concentration_field(
                dataset,
                column,
                radius,
                tau
            )
            average_concentration = self.volume_average(field, radius)
            mass_balance_residual = (
                average_concentration - expected_average
            )
            expected_mass_change = (
                expected_average[-1] - expected_average[0]
            )
            predicted_mass_change = (
                average_concentration[-1]
                - average_concentration[0]
            )

            if np.isclose(expected_mass_change, 0.0):
                relative_mass_change_error = np.nan
            else:
                relative_mass_change_error = abs(
                    predicted_mass_change - expected_mass_change
                ) / abs(expected_mass_change)

            checks[name] = {
                "mass_balance_passed": bool(
                    np.max(np.abs(mass_balance_residual))
                    <= mass_balance_tolerance
                ),
                "mass_balance_max_abs": float(
                    np.max(np.abs(mass_balance_residual))
                ),
                "mass_balance_rmse": float(
                    np.sqrt(np.mean(mass_balance_residual**2))
                ),
                "mass_balance_final_residual": float(
                    mass_balance_residual[-1]
                ),
                "relative_mass_change_error": float(
                    relative_mass_change_error
                ),
                "physical_bounds_passed": bool(
                    np.all((field >= 0.0) & (field <= 1.0))
                ),
                "minimum_concentration": float(np.min(field)),
                "maximum_concentration": float(np.max(field))
            }

        return checks

    @staticmethod
    def compute_hard_constraint_checks(
            hybrid_model,
            dataset,
            initial_concentration,
            device
            ):
        """Return initial-condition and centre-symmetry errors of the hybrid model."""
        initial_mask = np.isclose(
            dataset["tau"].to_numpy(),
            0.0
        )

        if not np.any(initial_mask):
            raise ValueError(
                "No tau = 0 samples were found in the corrected dataset."
            )

        initial_condition_max_error = float(
            np.max(
                np.abs(
                    dataset.loc[
                        initial_mask,
                        "concentration_hybrid"
                    ].to_numpy()
                    - initial_concentration
                )
            )
        )
        centre_tau = torch.tensor(
            np.sort(dataset["tau"].unique()),
            dtype=torch.float32,
            device=device
        )
        centre_coordinates = torch.column_stack(
            (
                torch.zeros_like(centre_tau),
                centre_tau
            )
        ).requires_grad_(True)
        centre_concentration = hybrid_model(
            centre_coordinates,
            initial_concentration
        )
        centre_derivatives = torch.autograd.grad(
            outputs=centre_concentration,
            inputs=centre_coordinates,
            grad_outputs=torch.ones_like(centre_concentration)
        )[0]
        centre_symmetry_max_error = float(
            centre_derivatives[:, 0]
            .abs()
            .max()
            .detach()
            .cpu()
        )

        return {
            "initial_condition_max_error": initial_condition_max_error,
            "centre_symmetry_max_error": centre_symmetry_max_error
        }

    @staticmethod
    def compute_surface_flux_check(
            hybrid_model,
            dataset,
            initial_concentration,
            device,
            alpha=1.0,
            activation_energy=35000.0,
            gas_constant=8.314462618,
            reference_temperature=298.15,
            ramp_time=300.0,
            ramp_down_start=3600.0
            ):
        """Evaluate the full-physics surface-flux residual.

        Inputs include the corrected dataset and diffusivity parameters.
        Returns global and phase-wise surface residual metrics.
        """
        surface_data = (
            dataset.loc[
                np.isclose(dataset["rho"], 1.0),
                [
                    "tau",
                    "time",
                    "temperature",
                    "dimensionless_flux"
                ]
            ]
            .sort_values("tau")
            .reset_index(drop=True)
        )

        if surface_data.empty:
            raise ValueError(
                "No rho = 1 samples were found for the surface-flux check."
            )

        tau = torch.tensor(
            surface_data["tau"].to_numpy(),
            dtype=torch.float32,
            device=device
        )
        surface_coordinates = torch.column_stack(
            (
                torch.ones_like(tau),
                tau
            )
        ).requires_grad_(True)
        surface_concentration = hybrid_model(
            surface_coordinates,
            initial_concentration
        )
        surface_derivatives = torch.autograd.grad(
            outputs=surface_concentration,
            inputs=surface_coordinates,
            grad_outputs=torch.ones_like(surface_concentration)
        )[0]
        surface_gradient = surface_derivatives[:, 0:1]

        temperature = torch.tensor(
            surface_data["temperature"].to_numpy(),
            dtype=torch.float32,
            device=device
        ).reshape(-1, 1)
        dimensionless_flux = torch.tensor(
            surface_data["dimensionless_flux"].to_numpy(),
            dtype=torch.float32,
            device=device
        ).reshape(-1, 1)
        normalized_diffusivity = (
            torch.exp(
                alpha
                * (
                    surface_concentration
                    - initial_concentration
                )
            )
            * torch.exp(
                -(activation_energy / gas_constant)
                * (
                    1.0 / temperature
                    - 1.0 / reference_temperature
                )
            )
        )
        reconstructed_flux = (
            -normalized_diffusivity * surface_gradient
        )
        surface_flux_residual = (
            reconstructed_flux - dimensionless_flux
        )

        residual = (
            surface_flux_residual
            .detach()
            .cpu()
            .numpy()
            .ravel()
        )
        time = surface_data["time"].to_numpy()
        metrics = {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "maximum_absolute_error": float(
                np.max(np.abs(residual))
            ),
            "final_residual": float(residual[-1]),
            "phases": {}
        }
        phase_intervals = {
            "ramp_up": (0.0, ramp_time),
            "constant_current": (ramp_time, ramp_down_start),
            "ramp_down": (
                ramp_down_start,
                ramp_down_start + ramp_time
            ),
            "rest": (
                ramp_down_start + ramp_time,
                float(time[-1])
            )
        }

        for phase_name, (start_time, end_time) in phase_intervals.items():
            phase_mask = (time >= start_time) & (time <= end_time)

            if np.any(phase_mask):
                phase_residual = residual[phase_mask]
                metrics["phases"][phase_name] = {
                    "rmse": float(
                        np.sqrt(np.mean(phase_residual**2))
                    ),
                    "maximum_absolute_error": float(
                        np.max(np.abs(phase_residual))
                    )
                }

        return metrics

    @staticmethod
    def compute_pde_consistency_checks(
            hybrid_model,
            dataset,
            initial_concentration,
            device,
            alpha=1.0,
            activation_energy=35000.0,
            gas_constant=8.314462618,
            reference_temperature=298.15,
            ramp_time=300.0,
            ramp_down_start=3600.0,
            batch_size=2048
            ):
        """Evaluate full and simplified PDE residuals on interior samples.

        Returns global/phase metrics, residual arrays, and numerical-regularity
        statistics for the corrected concentration field.
        """
        evaluation_data = (
            dataset.loc[
                (dataset["rho"] > 0.0)
                & (dataset["rho"] < 1.0)
                & (dataset["tau"] > 0.0),
                ["rho", "tau", "time", "temperature"]
            ]
            .sort_values(["tau", "rho"])
        )

        if evaluation_data.empty:
            raise ValueError(
                "No interior samples were found for the PDE check."
            )

        dataset_indices = evaluation_data.index.to_numpy()
        evaluation_data = evaluation_data.reset_index(drop=True)
        coordinates_numpy = evaluation_data[
            ["rho", "tau"]
        ].to_numpy()
        temperature_numpy = evaluation_data[
            "temperature"
        ].to_numpy()

        concentrations = []
        radial_gradients = []
        temporal_gradients = []
        radial_second_derivatives = []
        full_physics_residuals = []
        simplified_physics_residuals = []

        hybrid_model.eval()

        for start_index in range(
                0,
                len(evaluation_data),
                batch_size
                ):
            end_index = start_index + batch_size
            coordinates = torch.tensor(
                coordinates_numpy[start_index:end_index],
                dtype=torch.float32,
                device=device
            ).requires_grad_(True)
            temperature = torch.tensor(
                temperature_numpy[start_index:end_index],
                dtype=torch.float32,
                device=device
            ).reshape(-1, 1)
            concentration = hybrid_model(
                coordinates,
                initial_concentration
            )
            concentration_derivatives = torch.autograd.grad(
                outputs=concentration,
                inputs=coordinates,
                grad_outputs=torch.ones_like(concentration),
                create_graph=True
            )[0]
            radial_gradient = concentration_derivatives[:, 0:1]
            temporal_gradient = concentration_derivatives[:, 1:2]
            rho = coordinates[:, 0:1]
            normalized_diffusivity = (
                torch.exp(
                    alpha
                    * (concentration - initial_concentration)
                )
                * torch.exp(
                    -(activation_energy / gas_constant)
                    * (
                        1.0 / temperature
                        - 1.0 / reference_temperature
                    )
                )
            )
            weighted_radial_flux = (
                rho.square()
                * normalized_diffusivity
                * radial_gradient
            )
            weighted_flux_derivatives = torch.autograd.grad(
                outputs=weighted_radial_flux,
                inputs=coordinates,
                grad_outputs=torch.ones_like(weighted_radial_flux),
                retain_graph=True
            )[0]
            radial_second_derivative = torch.autograd.grad(
                outputs=radial_gradient,
                inputs=coordinates,
                grad_outputs=torch.ones_like(radial_gradient)
            )[0][:, 0:1]
            full_physics_residual = (
                temporal_gradient
                - weighted_flux_derivatives[:, 0:1]
                / rho.square()
            )
            simplified_physics_residual = (
                temporal_gradient
                - radial_second_derivative
                - (2.0 / rho) * radial_gradient
            )

            concentrations.append(concentration.detach().cpu())
            radial_gradients.append(radial_gradient.detach().cpu())
            temporal_gradients.append(temporal_gradient.detach().cpu())
            radial_second_derivatives.append(
                radial_second_derivative.detach().cpu()
            )
            full_physics_residuals.append(
                full_physics_residual.detach().cpu()
            )
            simplified_physics_residuals.append(
                simplified_physics_residual.detach().cpu()
            )

        concentration = torch.cat(concentrations).numpy().ravel()
        radial_gradient = torch.cat(
            radial_gradients
        ).numpy().ravel()
        temporal_gradient = torch.cat(
            temporal_gradients
        ).numpy().ravel()
        radial_second_derivative = torch.cat(
            radial_second_derivatives
        ).numpy().ravel()
        full_physics_residual = torch.cat(
            full_physics_residuals
        ).numpy().ravel()
        simplified_physics_residual = torch.cat(
            simplified_physics_residuals
        ).numpy().ravel()
        time = evaluation_data["time"].to_numpy()

        metrics = {
            "number_of_points": int(len(evaluation_data)),
            "dataset_indices": dataset_indices,
            "full_physics_residual": full_physics_residual,
            "simplified_physics_residual": simplified_physics_residual,
            "full_physics": {
                "rmse": float(
                    np.sqrt(np.mean(full_physics_residual**2))
                ),
                "maximum_absolute_error": float(
                    np.max(np.abs(full_physics_residual))
                ),
                "phases": {}
            },
            "simplified_physics": {
                "rmse": float(
                    np.sqrt(np.mean(simplified_physics_residual**2))
                ),
                "maximum_absolute_error": float(
                    np.max(np.abs(simplified_physics_residual))
                ),
                "phases": {}
            },
            "regularity": {
                "concentration_is_finite": bool(
                    np.isfinite(
                        dataset["concentration_hybrid"].to_numpy()
                    ).all()
                ),
                "derivatives_are_finite": bool(
                    np.isfinite(radial_gradient).all()
                    and np.isfinite(temporal_gradient).all()
                    and np.isfinite(radial_second_derivative).all()
                ),
                "pde_residuals_are_finite": bool(
                    np.isfinite(full_physics_residual).all()
                    and np.isfinite(simplified_physics_residual).all()
                ),
                "maximum_absolute_radial_gradient": float(
                    np.max(np.abs(radial_gradient))
                ),
                "maximum_absolute_temporal_gradient": float(
                    np.max(np.abs(temporal_gradient))
                ),
                "maximum_absolute_radial_second_derivative": float(
                    np.max(np.abs(radial_second_derivative))
                ),
                "interior_concentration_minimum": float(
                    np.min(concentration)
                ),
                "interior_concentration_maximum": float(
                    np.max(concentration)
                )
            }
        }
        phase_intervals = {
            "ramp_up": (0.0, ramp_time),
            "constant_current": (ramp_time, ramp_down_start),
            "ramp_down": (
                ramp_down_start,
                ramp_down_start + ramp_time
            ),
            "rest": (
                ramp_down_start + ramp_time,
                float(time[-1])
            )
        }

        for phase_name, (start_time, end_time) in phase_intervals.items():
            phase_mask = (time >= start_time) & (time <= end_time)

            if np.any(phase_mask):
                full_phase_residual = full_physics_residual[phase_mask]
                simplified_phase_residual = (
                    simplified_physics_residual[phase_mask]
                )
                metrics["full_physics"]["phases"][phase_name] = {
                    "rmse": float(
                        np.sqrt(np.mean(full_phase_residual**2))
                    ),
                    "maximum_absolute_error": float(
                        np.max(np.abs(full_phase_residual))
                    )
                }
                metrics["simplified_physics"]["phases"][phase_name] = {
                    "rmse": float(
                        np.sqrt(np.mean(simplified_phase_residual**2))
                    ),
                    "maximum_absolute_error": float(
                        np.max(np.abs(simplified_phase_residual))
                    )
                }

        return metrics

    @staticmethod
    def print_error_metrics(model_name, metrics):
        print("-" * 100)
        print(f"{model_name} ERROR METRICS AGAINST FVM")

        for quantity in ("field", "centre", "surface", "average"):
            quantity_metrics = metrics[quantity]
            print(
                f"{quantity.capitalize():8s} | "
                f"Relative L2: "
                f"{quantity_metrics['relative_l2']:.4e} | "
                f"RMSE: {quantity_metrics['rmse']:.4e} | "
                f"MAE: {quantity_metrics['mae']:.4e} | "
                "Max absolute error: "
                f"{quantity_metrics['maximum_absolute_error']:.4e}"
            )

        print(f"{model_name} FIELD ERROR BY OPERATING PHASE")

        for phase_name, phase_metrics in metrics["phases"].items():
            print(
                f"{phase_name:16s} | "
                f"Relative L2: "
                f"{phase_metrics['relative_l2']:.4e} | "
                f"RMSE: {phase_metrics['rmse']:.4e}"
            )
