import torch
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from time import perf_counter

class Testing(object):
    """Evaluate and reconstruct the FVM--FFN hybrid concentration model.

    It enforces a zero-volume-average correction, compares baseline and hybrid
    profiles, records timing/physics metrics, and exports corrected datasets.
    """

    def __init__(self,test_loader,radius,simplified_profiles):
        self.test_loader = test_loader
        self.radius = np.asarray(radius,dtype=float)

        if torch.is_tensor(simplified_profiles):
            simplified_profiles = simplified_profiles.detach().cpu().numpy()

        self.simplified_profiles = np.asarray(
            simplified_profiles,
            dtype=float
        )

        if self.radius.ndim != 1:
            raise ValueError("The radial grid must be one-dimensional.")

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

        model.eval()
        print(f"Testing...")

        with torch.no_grad():
            for x,y in self.test_loader:
                x = x.to(device)
                y = y.to(device)

                self.synchronize_device(device)
                inference_start = perf_counter()
                pred = model(x)
                pred = self.enforce_zero_average_correction(pred)
                self.synchronize_device(device)
                inference_time_seconds += perf_counter() - inference_start

                residual = torch.mean((y - pred).square())
                test_loss += residual.item() * x.shape[0]
                num_points += x.shape[0]

                test_predictions.append(pred.detach().cpu().numpy())
                test_targets.append(y.detach().cpu().numpy())

        test_loss = test_loss/num_points
        test_predictions = np.concatenate(test_predictions,axis=0)
        test_targets = np.concatenate(test_targets,axis=0)

        concentration_metrics = self.compute_concentration_metrics(
            test_targets,
            test_predictions
        )
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

    def enforce_zero_average_correction(self,corrections):
        """Project each radial correction onto the zero-volume-average subspace."""
        weights = torch.as_tensor(
            self.average_weights,
            dtype=corrections.dtype,
            device=corrections.device
        )

        correction_average = corrections @ weights

        return corrections - correction_average[:,None]

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
            baseline_errors @ self.average_weights
        )
        hybrid_average_metrics = self.compute_error_metrics(
            hybrid_errors @ self.average_weights
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
            "zero_average_correction_enforced": True
        }

        return metrics

    def compute_corrections(
            self,
            model,
            device,
            simplified_dataset
            ):
        """Predict mass-neutral corrections for a full simplified dataset.

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
            dtype=torch.float32,
            device=device,
        )

        model.eval()

        with torch.no_grad():
            corrections = model(features)
            corrections = self.enforce_zero_average_correction(corrections)

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

        average_concentration = new_concentrations @ self.average_weights
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
