import torch
import numpy as np

from pathlib import Path
from NeuralNetwork import FNN
from datetime import datetime
from time import perf_counter

class Processing(object):
    """Train the concentration-profile correction network.

    The processor minimizes radial residual MSE, selects the best validation
    state through early stopping, records timing, and saves a checkpoint.
    """

    def __init__(self,train_loader,val_loader,radius,inventory_offset):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.inventory_offset = float(inventory_offset)

        if not np.isfinite(self.inventory_offset):
            raise ValueError("The inventory offset must be finite.")

        self.training_time_seconds = 0.0
        self.maximum_training_epochs = 0
        self.completed_training_epochs = 0
        self.ending_training_epoch = 0
        self.early_stopped = False
        self.checkpoint_path = None

        self.average_weights = self.compute_average_weights(radius)

    def train(
            self,
            lr,
            epochs,
            early_stopping_patience,
            input_dim,
            hidden_dim,
            out_dim,
            device,
            save_model_checkpoint=True
            ):
        """Fit and return the best validation residual-profile network."""

        print('-'*100)
        print("Training...")

        model = FNN(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=out_dim)

        model.to(device)
        optimizer = torch.optim.Adam(lr=lr,params=model.parameters())

        patience = 0
        best_model_state = None
        best_val_loss = float("inf")
        best_epoch = 0

        self.maximum_training_epochs = epochs
        self.completed_training_epochs = 0
        self.ending_training_epoch = 0
        self.early_stopped = False
        training_start = perf_counter()

        for epoch in range(epochs):
            self.completed_training_epochs = epoch + 1
            self.ending_training_epoch = epoch + 1

            model.train()
            loss_sum = 0.0
            num_points = 0

            for x,y in self.train_loader:
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                pred, _, _ = self.apply_inventory_constraint(pred)
                residual = torch.mean((y - pred).square())

                residual.backward()
                optimizer.step()

                loss_sum += residual.item() * x.shape[0]
                num_points += x.shape[0]

            train_loss = loss_sum / num_points
            val_loss = self.validate(model,device=device)

            print(
                f"Epoch {epoch + 1:5d}/{epochs} | "
                f"Total_train_loss: {train_loss:.4e} | "
                f"Total_val_loss: {val_loss:.4e} | "
                f"Inventory offset: {self.inventory_offset:.4e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} |"
                f"OPT: ADAM"
                )
            print("-" * 100)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = {name : value.detach().clone()
                                    for name,value in model.state_dict().items()
                                    }
                patience = 0

            else:
                patience += 1
                if patience >= early_stopping_patience:
                    self.early_stopped = True
                    print("Plateau reached.")
                    break

        self.training_time_seconds = perf_counter() - training_start

        print("Training ended.")
        print(
            f"Best validation loss: {best_val_loss:.4e} "
            f"(epoch: {best_epoch}) | "
            f"Inventory offset: {self.inventory_offset:.4e}"
        )
        print("-"*100)
        model.load_state_dict(best_model_state)

        training_metrics = {
            "inventory_offset": float(self.inventory_offset),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_rmse": float(np.sqrt(best_val_loss))
        }

        if save_model_checkpoint:
            self.checkpoint_path = self.save_checkpoint(
                model,
                best_val_loss,
                optimizer_name="Adam",
                lr=lr,
                training_metrics=training_metrics
            )

        return model,training_metrics

    def validate(self,model,device):
        """Return sample-weighted validation MSE for ``model``."""

        model.eval()
        val_loss_sum = 0.0
        num_points = 0

        with torch.no_grad():
            for x,y in self.val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                pred, _, _ = self.apply_inventory_constraint(pred)
                residual = torch.mean((y - pred).square())

                val_loss_sum += residual.item() * x.shape[0]
                num_points += x.shape[0]

        return val_loss_sum / num_points

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

    @staticmethod
    def estimate_inventory_offset(targets,radius):
        """Estimate the constant average correction from training targets."""
        if not torch.is_tensor(targets):
            targets = torch.as_tensor(targets,dtype=torch.float32)

        weights = torch.as_tensor(
            Processing.compute_average_weights(radius),
            dtype=targets.dtype,
            device=targets.device
        )

        if targets.ndim != 2 or targets.shape[1] != weights.numel():
            raise ValueError(
                "Targets must have shape (number_samples, number_radius)."
            )

        return float(torch.mean(targets @ weights).item())

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

        return final_corrections,raw_average,constant_average

    def save_checkpoint(
            self,
            model,
            val_loss,
            optimizer_name,
            lr,
            training_metrics,
            output_name="checkpoint.pth",
            output_folder="./Results/Models"
            ):
        """Save model weights and training metadata; return the checkpoint path."""

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True,exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename_path = Path(output_name)
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
            "val_loss": float(val_loss),
            "optimizer": optimizer_name,
            "lr": lr,
            "inventory_offset": float(self.inventory_offset),
            "constraint": "constant_volume_average",
            "training_metrics": training_metrics
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print('-'*100)
        print(
            f"Model checkpoint saved to: "
            f"{checkpoint_path.resolve()}"
        )
        print('-'*100)

        return checkpoint_path
