import torch
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from time import perf_counter

class Testing(object):
    """Evaluate and reconstruct the FVM--FFN hybrid concentration model.

    It fixes the volume-average correction, compares baseline and hybrid
    profiles, records timing/physics metrics, and exports corrected datasets.
    """

    def __init__(
            self,
            test_loader,
            radius,
            simplified_profiles,
            test_tau,
            inventory_offset
            ):
        self.test_loader = test_loader
        self.radius = np.asarray(radius,dtype=float)
        self.test_tau = np.asarray(test_tau,dtype=float)
        self.inventory_offset = float(inventory_offset)
        self.concentration_metrics = None

        if not np.isfinite(self.inventory_offset):
            raise ValueError("The inventory offset must be finite.")

        if torch.is_tensor(simplified_profiles):
            simplified_profiles = simplified_profiles.detach().cpu().numpy()

        self.simplified_profiles = np.asarray(
            simplified_profiles,
            dtype=float
        )

        if self.radius.ndim != 1:
            raise ValueError("The radial grid must be one-dimensional.")

        if self.test_tau.ndim != 1:
            raise ValueError("The test time grid must be one-dimensional.")

        if len(self.test_tau) != len(self.test_loader.dataset):
            raise ValueError(
                "The test time grid and test dataset must have the same length."
            )

        if self.simplified_profiles.shape != (
                len(self.test_loader.dataset),
                len(self.radius)
                ):
            raise ValueError(
                "The simplified test profiles have an invalid shape."
            )

        self.average_weights = self.compute_average_weights(self.radius)

    def test(
            self,
            model,
            device,
            training_time_seconds,
            maximum_training_epochs,
            completed_training_epochs,
            ending_training_epoch,
            early_stopped,
            output_folder="./Results",
            filename="test_history.csv",
            variant_number=None
            ):
        """Evaluate held-out profile corrections and save aggregate metrics.

        Inputs include model, execution metadata, and output settings. Returns
        the corrected-field mean-squared error.
        """
        
        test_loss = 0.0
        num_points = 0
        inference_time_seconds = 0.0
        test_predictions = []
        test_targets = []
        test_average_corrections = []

        model.eval()
        print(f"Testing...")

        with torch.no_grad():
            for x,y in self.test_loader:
                x = x.to(device)
                y = y.to(device)

                self.synchronize_device(device)
                inference_start = perf_counter()
                pred = model(x)
                pred, _, average_correction = self.apply_inventory_constraint(
                    pred
                )
                self.synchronize_device(device)
                inference_time_seconds += perf_counter() - inference_start

                residual = torch.mean((y - pred).square())
                test_loss += residual.item() * x.shape[0]
                num_points += x.shape[0]

                test_predictions.append(pred.detach().cpu().numpy())
                test_targets.append(y.detach().cpu().numpy())
                test_average_corrections.append(
                    average_correction.detach().cpu().numpy()
                )

        test_loss = test_loss/num_points
        test_predictions = np.concatenate(test_predictions,axis=0)
        test_targets = np.concatenate(test_targets,axis=0)
        test_average_corrections = np.concatenate(
            test_average_corrections,
            axis=0
        )

        concentration_metrics = self.compute_concentration_metrics(
            test_targets,
            test_predictions
        )
        concentration_metrics["test_constant_average_temporal_drift"] = (
            self.compute_temporal_drift(test_average_corrections)
        )
        concentration_metrics["test_inventory_constraint_max_error"] = float(
            np.max(
                np.abs(test_average_corrections - self.inventory_offset)
            )
        )
        self.concentration_metrics = concentration_metrics.copy()
        test_loss = concentration_metrics["hybrid_concentration_mse"]

        print(f"Test_loss: {test_loss}")
        print(
            "Concentration RMSE | "
            f"Simplified: {concentration_metrics['baseline_concentration_rmse']:.6e} | "
            f"Hybrid: {concentration_metrics['hybrid_concentration_rmse']:.6e} | "
            f"RER: {concentration_metrics['concentration_rmse_reduction_percent']:.2f}%"
        )
        print(
            "Surface concentration RMSE | "
            f"Simplified: {concentration_metrics['baseline_surface_concentration_rmse']:.6e} | "
            f"Hybrid: {concentration_metrics['hybrid_surface_concentration_rmse']:.6e}"
        )
        print(
            "Average concentration RMSE | "
            f"Simplified: {concentration_metrics['baseline_average_concentration_rmse']:.6e} | "
            f"Hybrid: {concentration_metrics['hybrid_average_concentration_rmse']:.6e}"
        )
        print(
            "Constant-average temporal drift: "
            f"{concentration_metrics['test_constant_average_temporal_drift']:.6e}"
        )
        print(
            "Inventory constraint maximum error: "
            f"{concentration_metrics['test_inventory_constraint_max_error']:.6e}"
        )
        print(f"Inference time: {inference_time_seconds:.6f} s")
        print("-"*100)

        self.save_results(
            test_loss,
            device,
            training_time_seconds,
            inference_time_seconds,
            num_points,
            maximum_training_epochs,
            completed_training_epochs,
            ending_training_epoch,
            early_stopped,
            concentration_metrics,
            output_folder=output_folder,
            filename=filename,
            variant_number=variant_number
        )

        return test_loss

    @staticmethod
    def synchronize_device(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    @staticmethod
    def compute_average_weights(radius):
        """Return normalized trapezoidal weights for a spherical volume average."""
        radius = np.asarray(radius,dtype=float)

        if radius.ndim != 1 or len(radius) < 2:
            raise ValueError(
                "The radial grid must contain at least two points."
            )

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

    def apply_inventory_constraint(self,corrections):
        weights = torch.as_tensor(
            self.average_weights,
            dtype=corrections.dtype,
            device=corrections.device
        )

        if corrections.ndim != 2:
            raise ValueError(
                "Corrections must have shape (batch_size, n_radius)."
            )

        if corrections.shape[1] != weights.numel():
            raise ValueError(
                "Corrections and radial weights have incompatible dimensions."
            )

        raw_average = corrections @ weights
        constant_average = torch.full_like(
            raw_average,
            self.inventory_offset
        )
        final_corrections = (
            corrections
            - raw_average[:,None]
            + constant_average[:,None]
        )
        final_average = final_corrections @ weights

        return final_corrections,raw_average,final_average

    def compute_temporal_drift(self,average_corrections):
        """Return maximum average-correction drift from the earliest test time."""
        average_corrections = np.asarray(average_corrections,dtype=float)

        if average_corrections.shape != self.test_tau.shape:
            raise ValueError(
                "Average corrections and test times must have the same shape."
            )

        time_order = np.argsort(self.test_tau)
        ordered_averages = average_corrections[time_order]

        return float(
            np.max(np.abs(ordered_averages - ordered_averages[0]))
        )

    @staticmethod
    def compute_error_metrics(errors):
        errors = np.asarray(errors,dtype=float)
        absolute_errors = np.abs(errors)

        return {
            "mse": float(np.mean(errors ** 2)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(absolute_errors)),
            "max_abs_error": float(np.max(absolute_errors))
        }

    @staticmethod
    def compute_error_reduction(baseline_error,hybrid_error):
        if baseline_error == 0.0:
            return np.nan

        return float(
            (baseline_error - hybrid_error) / baseline_error * 100.0
        )

    def compute_concentration_metrics(self,targets,predictions):
        """Return baseline/hybrid field, surface, average, and bounds metrics."""
        targets = np.asarray(targets,dtype=float)
        predictions = np.asarray(predictions,dtype=float)

        if targets.shape != predictions.shape:
            raise ValueError(
                "Targets and predictions must have the same shape."
            )

        baseline_errors = -targets
        hybrid_errors = predictions - targets

        baseline_metrics = self.compute_error_metrics(baseline_errors)
        hybrid_metrics = self.compute_error_metrics(hybrid_errors)

        baseline_surface_metrics = self.compute_error_metrics(
            baseline_errors[:,-1]
        )
        hybrid_surface_metrics = self.compute_error_metrics(
            hybrid_errors[:,-1]
        )

        baseline_average_metrics = self.compute_error_metrics(
            np.sum(
                baseline_errors * self.average_weights[None,:],
                axis=1
            )
        )
        hybrid_average_metrics = self.compute_error_metrics(
            np.sum(
                hybrid_errors * self.average_weights[None,:],
                axis=1
            )
        )

        hybrid_profiles = self.simplified_profiles + predictions
        physical_bounds_violations = np.count_nonzero(
            (hybrid_profiles < 0.0) | (hybrid_profiles > 1.0)
        )

        metrics = {
            "baseline_concentration_mse": baseline_metrics["mse"],
            "baseline_concentration_rmse": baseline_metrics["rmse"],
            "baseline_concentration_mae": baseline_metrics["mae"],
            "baseline_concentration_max_abs_error": baseline_metrics["max_abs_error"],
            "hybrid_concentration_mse": hybrid_metrics["mse"],
            "hybrid_concentration_rmse": hybrid_metrics["rmse"],
            "hybrid_concentration_mae": hybrid_metrics["mae"],
            "hybrid_concentration_max_abs_error": hybrid_metrics["max_abs_error"],
            "concentration_rmse_reduction_percent": self.compute_error_reduction(
                baseline_metrics["rmse"],
                hybrid_metrics["rmse"]
            ),
            "baseline_surface_concentration_rmse": baseline_surface_metrics["rmse"],
            "baseline_surface_concentration_mae": baseline_surface_metrics["mae"],
            "hybrid_surface_concentration_rmse": hybrid_surface_metrics["rmse"],
            "hybrid_surface_concentration_mae": hybrid_surface_metrics["mae"],
            "surface_concentration_rmse_reduction_percent": self.compute_error_reduction(
                baseline_surface_metrics["rmse"],
                hybrid_surface_metrics["rmse"]
            ),
            "baseline_average_concentration_rmse": baseline_average_metrics["rmse"],
            "baseline_average_concentration_mae": baseline_average_metrics["mae"],
            "hybrid_average_concentration_rmse": hybrid_average_metrics["rmse"],
            "hybrid_average_concentration_mae": hybrid_average_metrics["mae"],
            "hybrid_concentration_min": float(np.min(hybrid_profiles)),
            "hybrid_concentration_max": float(np.max(hybrid_profiles)),
            "hybrid_physical_bounds_violations": int(physical_bounds_violations),
            "hybrid_physical_bounds_violation_fraction": float(
                physical_bounds_violations / hybrid_profiles.size
            ),
            "constant_average_correction_enforced": True,
            "inventory_offset": self.inventory_offset
        }

        return metrics

    def compute_corrections(
            self,
            model,
            device,
            simplified_dataset
            ):
        """Predict constant-average corrections for a full simplified dataset.

        Returns tensors containing simplified concentrations and corrections.
        """

        profiles = (
            simplified_dataset
            .pivot(index="tau", columns="rho", values="concentration")
            .sort_index()
            .sort_index(axis=1)
        )

        states = (
            simplified_dataset
            .drop_duplicates("tau")
            .set_index("tau")
            .sort_index()
            .loc[profiles.index]
        )

        features = torch.tensor(
            states[
                [
                    "average_concentration",
                    "surface_concentration",
                    "current",
                ]
            ].to_numpy(),
            dtype=torch.float32,
            device=device,
        )

        simplified_concentrations = torch.tensor(
            profiles.to_numpy(),
            dtype=torch.float64
        )

        model.eval()

        with torch.no_grad():
            corrections = model(features)
            corrections, _, _ = self.apply_inventory_constraint(corrections)
            corrections = corrections.detach().cpu().to(dtype=torch.float64)

        return simplified_concentrations, corrections

    def save_results(
            self,
            test_loss,
            device,
            training_time_seconds,
            inference_time_seconds,
            number_test_points,
            maximum_training_epochs,
            completed_training_epochs,
            ending_training_epoch,
            early_stopped,
            concentration_metrics,
            output_folder="./Results",
            filename="test_history.csv",
            variant_number=None
            ):
        """Append one hybrid test record and return the results CSV path."""
        
        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
            )
        
        results_path = output_folder / filename

        test_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "test_loss": test_loss,
            "training_time_seconds": training_time_seconds,
            "test_inference_time_seconds": inference_time_seconds,
            "test_inference_time_seconds_per_sample": (inference_time_seconds / number_test_points),
            "maximum_training_epochs": maximum_training_epochs,
            "completed_training_epochs": completed_training_epochs,
            "ending_training_epoch": ending_training_epoch,
            "early_stopped": early_stopped,
            "variant_number":variant_number
            }

        test_record.update(concentration_metrics)

        results_frame = pd.DataFrame([test_record])

        if results_path.exists():
            previous_results = pd.read_csv(results_path)
            results_frame = pd.concat(
                [previous_results,results_frame],
                ignore_index=True
            )

        results_frame.to_csv(results_path,index=False)

        return results_path

    def generate_new_dataset(
            self,
            simplified_dataset,
            new_concentrations,
            output_path,
            filename="corrected_dataset.csv",
            ):
        """Replace concentration-derived columns and save a corrected grid.

        Inputs are a complete simplified dataset and corrected concentration
        matrix. Returns the generated CSV path.
        """

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True,parents=True)

        results_path = output_path / filename
        new_frame = (
            simplified_dataset
            .sort_values(["tau", "rho"])
            .reset_index(drop=True)
            .copy()
        )

        if torch.is_tensor(new_concentrations):
            new_concentrations = (
                new_concentrations
                .detach()
                .cpu()
                .numpy()
            )

        new_concentrations = np.asarray(new_concentrations)

        tau = np.sort(new_frame["tau"].unique())
        radius = np.sort(new_frame["rho"].unique())
        expected_shape = (len(tau),len(radius))

        if new_frame.duplicated(["tau","rho"]).any():
            raise ValueError(
                "The dataset contains duplicated space-time points."
            )

        if len(new_frame) != expected_shape[0] * expected_shape[1]:
            raise ValueError(
                "The dataset must contain a complete rectangular grid."
            )

        if new_concentrations.shape == (len(new_frame),):
            new_concentrations = new_concentrations.reshape(expected_shape)

        if new_concentrations.shape != expected_shape:
            raise ValueError(
                "The corrected concentration field must have shape "
                f"{expected_shape}, received {new_concentrations.shape}."
            )

        if not np.allclose(radius,self.radius):
            raise ValueError(
                "The dataset and correction radial grids do not match."
            )

        average_concentration = np.sum(
            new_concentrations * self.average_weights[None,:],
            axis=1
        )
        surface_concentration = new_concentrations[:,-1]

        new_frame["concentration"] = new_concentrations.reshape(-1)
        new_frame["average_concentration"] = np.repeat(
            average_concentration,
            len(radius)
        )
        new_frame["surface_concentration"] = np.repeat(
            surface_concentration,
            len(radius)
        )

        new_frame.to_csv(results_path,index=False)

        return results_path
