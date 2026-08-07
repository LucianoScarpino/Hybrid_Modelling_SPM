import torch
import pandas as pd


from datetime import datetime
from pathlib import Path
from time import perf_counter

class Testing(object):
    """Evaluate standalone PINN and residual-correction models.

    The tester reports held-out concentration and physics errors, records
    inference timing, and appends reproducible CSV histories.
    """

    def __init__(
            self,
            d_test_loader=None, 
            f_test_loader=None, 
            b_test_loader=None
            ):
        
        self.data_loader = d_test_loader
        self.pde_loader = f_test_loader
        self.boundary_loader = b_test_loader

    @staticmethod
    def _synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()

    def test(
            self,
            model,
            init_concentration,
            flux
            ):
        """Test PINN accuracy plus PDE and boundary consistency.

        Returns relative field/surface errors, physics RMSE values, and timing.
        """
        model.eval()
        model_device = next(model.parameters()).device

        concentration_error_squared_sum = 0.0
        concentration_reference_squared_sum = 0.0
        surface_error_squared_sum = 0.0
        surface_reference_squared_sum = 0.0
        pde_residual_squared_sum = 0.0
        boundary_residual_squared_sum = 0.0

        number_data_points = 0
        number_surface_points = 0
        number_collocation_points = 0
        number_boundary_points = 0

        inference_time = 0.0
        warmup_completed = False

        # Concentration errors against the finite-volume reference solution.
        with torch.inference_mode():
            for x_data, y_data in self.data_loader:
                x_data = x_data.to(model_device)
                y_data = y_data.to(model_device)

                if not warmup_completed:
                    model(x_data, init_concentration)
                    self._synchronize(model_device)
                    warmup_completed = True

                self._synchronize(model_device)
                start_time = perf_counter()
                concentration_prediction = model(
                    x_data,
                    init_concentration
                    )
                self._synchronize(model_device)
                inference_time += perf_counter() - start_time

                concentration_error_squared_sum += torch.sum(
                    (concentration_prediction - y_data).square()
                    ).item()
                concentration_reference_squared_sum += torch.sum(
                    y_data.square()
                    ).item()
                number_data_points += x_data.shape[0]

                surface_mask = torch.isclose(
                    x_data[:, 0],
                    torch.ones_like(x_data[:, 0])
                    )

                if surface_mask.any().item():
                    surface_prediction = concentration_prediction[surface_mask]
                    surface_reference = y_data[surface_mask]

                    surface_error_squared_sum += torch.sum(
                        (surface_prediction - surface_reference).square()
                        ).item()
                    surface_reference_squared_sum += torch.sum(
                        surface_reference.square()
                        ).item()
                    number_surface_points += surface_reference.shape[0]

        # PDE residual on independent collocation test points.
        for (x_collocation_batch,) in self.pde_loader:
            x_collocation = (
                x_collocation_batch
                .to(model_device)
                .detach()
                .clone()
                .requires_grad_(True)
                )

            concentration_collocation = model(
                x_collocation,
                init_concentration
                )

            first_derivatives = torch.autograd.grad(
                outputs=concentration_collocation,
                inputs=x_collocation,
                grad_outputs=torch.ones_like(concentration_collocation),
                create_graph=True
                )[0]

            dC_drho = first_derivatives[:, :1]
            dC_dtau = first_derivatives[:, 1:2]

            second_derivatives = torch.autograd.grad(
                outputs=dC_drho,
                inputs=x_collocation,
                grad_outputs=torch.ones_like(dC_drho)
                )[0]

            d2C_drho2 = second_derivatives[:, :1]
            rho = x_collocation[:, :1]

            pde_residual = (
                dC_dtau
                - d2C_drho2
                - (2.0 / rho) * dC_drho
                )

            pde_residual_squared_sum += torch.sum(
                pde_residual.square()
                ).item()
            number_collocation_points += x_collocation.shape[0]

        # Outer Neumann boundary residual on independent boundary test points.
        for (x_boundary_batch,) in self.boundary_loader:
            x_boundary = (
                x_boundary_batch
                .to(model_device)
                .detach()
                .clone()
                .requires_grad_(True)
                )

            concentration_boundary = model(
                x_boundary,
                init_concentration
                )

            boundary_derivatives = torch.autograd.grad(
                outputs=concentration_boundary,
                inputs=x_boundary,
                grad_outputs=torch.ones_like(concentration_boundary)
                )[0]

            dC_drho_boundary = boundary_derivatives[:, :1]
            boundary_residual = -dC_drho_boundary - flux

            boundary_residual_squared_sum += torch.sum(
                boundary_residual.square()
                ).item()
            number_boundary_points += x_boundary.shape[0]

        if number_data_points == 0:
            raise ValueError("The data test loader is empty.")
        if number_surface_points == 0:
            raise ValueError(
                "No rho = 1 samples were found in the data test loader."
                )
        if number_collocation_points == 0:
            raise ValueError("The collocation test loader is empty.")
        if number_boundary_points == 0:
            raise ValueError("The boundary test loader is empty.")
        if concentration_reference_squared_sum == 0.0:
            raise ValueError("The concentration reference norm is zero.")
        if surface_reference_squared_sum == 0.0:
            raise ValueError("The surface concentration reference norm is zero.")

        relative_concentration_error = (
            concentration_error_squared_sum
            / concentration_reference_squared_sum
            ) ** 0.5
        relative_surface_error = (
            surface_error_squared_sum
            / surface_reference_squared_sum
            ) ** 0.5
        pde_rmse = (
            pde_residual_squared_sum
            / number_collocation_points
            ) ** 0.5
        boundary_rmse = (
            boundary_residual_squared_sum
            / number_boundary_points
            ) ** 0.5

        return {
            "relative_concentration_error": relative_concentration_error,
            "relative_surface_error": relative_surface_error,
            "boundary_rmse": boundary_rmse,
            "pde_rmse": pde_rmse,
            "inference_time_seconds": inference_time,
            "inference_time_seconds_per_sample": (
                inference_time / number_data_points
                )
            }

    def save_test_results(
            self,
            metrics,
            device,
            initial_concentration,
            flux,
            output_folder="./Results",
            filename="test_history.csv"
            ):
        """Append one standalone-PINN test record and return its CSV path."""
        
        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
            )

        results_path = output_folder / filename

        test_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "initial_concentration": float(initial_concentration),
            "flux": float(flux),
            **metrics
            }

        results_frame = pd.DataFrame([test_record])

        results_frame.to_csv(
            results_path,
            mode="a",
            header=not results_path.exists(),
            index=False
            )

        return results_path

    def test_residuals(
            self,
            residual_model,
            residual_loader
            ): 
        """Test residual prediction and reconstructed Hybrid-PINN accuracy.

        Returns the metric dictionary together with predicted and target
        correction tensors.
        """
        
        print("Testing Residual Learner...")

        residual_model.eval()
        model_device = next(residual_model.parameters()).device

        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        reference_squared_sum = 0.0
        num_points = 0

        residual_predictions = []
        residual_targets = []

        with torch.inference_mode():
            for X, y in residual_loader:
                X = X.to(model_device)
                y = y.to(model_device)

                predicted_residual = residual_model(X)

                if predicted_residual.shape != y.shape:
                    raise ValueError(
                        f"Shape incompatibili: "
                        f"pred={predicted_residual.shape}, target={y.shape}"
                    )

                error = predicted_residual - y

                # C_hybrid = C_PINN + residual prediction
                concentration_pinn = X[:, 2:3]
                concentration_reference = concentration_pinn + y

                squared_error_sum += torch.sum(error.square()).item()
                absolute_error_sum += torch.sum(error.abs()).item()
                reference_squared_sum += torch.sum(
                    concentration_reference.square()
                ).item()

                num_points += X.shape[0]

                residual_predictions.append(
                    predicted_residual.cpu()
                )
                residual_targets.append(y.cpu())

        if num_points == 0:
            raise ValueError("The residual test loader is empty.")

        if reference_squared_sum == 0.0:
            raise ValueError(
                "The reference concentration norm is zero."
            )

        mse = squared_error_sum / num_points
        rmse = mse ** 0.5
        mae = absolute_error_sum / num_points
        relative_hybrid_error = (
            squared_error_sum / reference_squared_sum
        ) ** 0.5

        metrics = {
            "residual_mse": mse,
            "residual_rmse": rmse,
            "residual_mae": mae,
            "relative_hybrid_concentration_error": relative_hybrid_error,
        }

        predictions = torch.cat(residual_predictions, dim=0)
        targets = torch.cat(residual_targets, dim=0)

        print(
            f"Residual RMSE: {rmse:.6e} | "
            f"Residual MAE: {mae:.6e} | "
            f"Hybrid relative L2: {relative_hybrid_error:.6e}"
        )
        print("-" * 100)

        self.save_residual_test(model_device,rmse,mae,relative_hybrid_error)

        return metrics, predictions, targets

    def save_residual_test(
            self,
            device,
            res_RMSE,
            res_MAE,
            Hybrid_rel_L2,
            output_folder="./Results",
            filename="residual_test_history.csv"
            ):
        """Append one residual-model test record and return its CSV path."""

        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
            )

        results_path = output_folder / filename

        test_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "residual_RMSE": float(res_RMSE),
            "residual_MAE": float(res_MAE),
            "hybrid_rel_L2": float(Hybrid_rel_L2),
            }

        results_frame = pd.DataFrame([test_record])

        results_frame.to_csv(
            results_path,
            mode="a",
            header=not results_path.exists(),
            index=False
            )

        return results_path
