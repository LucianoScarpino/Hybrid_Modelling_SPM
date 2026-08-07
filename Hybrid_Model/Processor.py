import torch
import numpy as np
import os

from pathlib import Path
from NeuralNetwork import FNN
from datetime import datetime
from time import perf_counter

class Processing(object):
    """Train the concentration-profile correction network.

    The processor minimizes radial residual MSE, selects the best validation
    state through early stopping, records timing, and saves a checkpoint.
    """

    def __init__(self,train_loader,val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_time_seconds = 0.0
        self.maximum_training_epochs = 0
        self.completed_training_epochs = 0
        self.ending_training_epoch = 0
        self.early_stopped = False

    def train(
            self,
            lr,
            epochs,
            early_stopping_patience,
            input_dim,
            hidden_dim,
            out_dim,
            device
            ):
        """Fit and return the best validation residual-profile network."""

        print("Training...")
        model = FNN(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=out_dim)
        model.to(device)
        optimizer = torch.optim.Adam(lr=lr,params=model.parameters())

        patience = 0
        best_model_state = None
        best_val_loss = float("inf")
        train_loss = 0.0

        self.maximum_training_epochs = epochs
        self.completed_training_epochs = 0
        self.ending_training_epoch = 0
        self.early_stopped = False
        training_start = perf_counter()

        for epoch in range(epochs):
            self.completed_training_epochs = epoch + 1
            self.ending_training_epoch = epoch + 1

            model.train ()
            loss_sum = 0.0
            num_points = 0

            for x,y in self.train_loader:
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                residual = torch.mean((y - pred).square())

                residual.backward()
                optimizer.step()

                loss_sum += residual.item() * x.shape[0]
                num_points += x.shape[0]

            train_loss = loss_sum/num_points

            val_loss = self.validate(model,device=device)

            print(
                f"Epoch {epoch + 1:5d}/{epochs} | "
                f"Total_train_loss: {train_loss:.4e} | "
                f"Total_val_loss: {val_loss:.4e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} |"
                f"OPT: ADAM"
                )
            print("-" * 100)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
        print("-"*100)
        model.load_state_dict(best_model_state)
        self.save_checkpoint(model,best_val_loss,optimizer_name="Adam",lr=lr)

        return model


    def validate(self,model,device):
        """Return sample-weighted validation MSE for ``model``."""
        model.eval()
        val_loss = 0.0
        num_points = 0

        with torch.no_grad():
            for x,y in self.val_loader:
                x = x.to(device)
                y = y.to(device)
                
                pred = model(x)
                residual = torch.mean((y - pred).square())
                val_loss += residual.item() * x.shape[0]
                num_points += x.shape[0]

        return val_loss/num_points

    def save_checkpoint(
            self,
            model,
            val_loss,
            optimizer_name,
            lr,
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
            "lr": lr
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
