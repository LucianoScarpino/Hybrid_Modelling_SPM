import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import TensorDataset,DataLoader
from scipy.stats import qmc
from sklearn.model_selection import train_test_split

class DatasetGenerator(object):
    """Prepare supervised, collocation, boundary, and residual-learning data.

    The class reads reference simulations, applies reproducible splits, and
    returns PyTorch loaders for the standalone and post-training PINN stages.
    """

    def __init__(self,data_location,length=20301):
        self.data = pd.read_csv(data_location)
        self.N = length

        os.makedirs("Data",exist_ok=True)

    def get_reference_dataset(self,reference_data_folder="./Data/full_simulation_dataset.csv"):
        """Load ``rho``, ``tau``, and concentration from the FVM dataset."""
        reference_data = pd.read_csv(reference_data_folder)

        reference_dataset = reference_data[["rho","tau","concentration"]].copy()

        return reference_dataset

    def generate_collocation_points(self):
        """Extract interior space--time coordinates and save them as CSV."""
        new_frame = self.data.loc[
            (self.data["rho"] > 0.0)
            & (self.data["rho"] < 1.0)
            & (self.data["tau"] > 0.0),
            ["rho", "tau"],
            ]

        [['rho','tau']]
        new_frame.to_csv("./Data/collocation_dataset.csv",index=False)

    def generate_boundary_points(self,tauf=1.007):
        """Generate Sobol time samples on the particle surface."""
        sobol = qmc.Sobol(d=1)
        rho = np.ones((self.N,1))
        taut = tauf * sobol.random(self.N)
        data = np.hstack([rho,taut])

        frame = pd.DataFrame(data=data,columns=['rho=1','tau'])
        frame.to_csv("./Data/boundary_dataset.csv",index=False)

    def generate_residual_dataset(
        self,
        pinn_model,
        reference_loader,
        initial_concentration
        ):
        """Build inputs and targets for residual learning.

        Inputs are reference batches and a frozen PINN; the returned
        ``TensorDataset`` contains ``(rho, tau, C_PINN)`` and ``C_FVM-C_PINN``.
        """

        pinn_model.eval()
        model_device = next(pinn_model.parameters()).device

        residual_inputs = []
        residual_targets = []

        with torch.inference_mode():
            for coordinates, concentration_reference in reference_loader:
                coordinates = coordinates.to(model_device)
                concentration_reference = concentration_reference.to(model_device)

                concentration_pinn = pinn_model(coordinates,initial_concentration)

                X_residual = torch.cat((coordinates,concentration_pinn),dim=1)
                y_residual = (concentration_reference - concentration_pinn)

                residual_inputs.append(X_residual.cpu())
                residual_targets.append(y_residual.cpu())

        X_residual = torch.cat(residual_inputs, dim=0)
        y_residual = torch.cat(residual_targets, dim=0)

        return TensorDataset(X_residual, y_residual)

    def generate_residual_loaders(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=512
        ):
        """Wrap pre-split residual datasets in train, validation, and test loaders."""

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def split_dataset(self,dataset):
        """Return a reproducible 70/15/15 random split of ``dataset``."""
        train,temporary = train_test_split(dataset,
                                           test_size=0.30,
                                           random_state=26,
                                           shuffle=True)

        validation,test = train_test_split(temporary,
                                           test_size=0.50,
                                           random_state=26,
                                           shuffle=True)

        return train,validation,test

    def generate_dataloaders(self,dataset,labels=True,batch_size=512):
        """Convert a split numeric dataset into PyTorch loaders.

        ``labels`` selects supervised ``(X, y)`` or coordinate-only batches.
        Returns train, validation, and test loaders.
        """
        X_train, X_val, X_test = self.split_dataset(dataset)

        if labels:
            x_train,y_train = torch.tensor(X_train[:,:2],dtype=torch.float32), torch.tensor(X_train[:,2:3],dtype=torch.float32)
            x_val,y_val = torch.tensor(X_val[:,:2],dtype=torch.float32), torch.tensor(X_val[:,2:3],dtype=torch.float32)
            x_test,y_test = torch.tensor(X_test[:,:2],dtype=torch.float32), torch.tensor(X_test[:,2:3],dtype=torch.float32)

            if batch_size is not None:
                train_loader = DataLoader(
                    TensorDataset(x_train,y_train),
                    batch_size=batch_size,
                    shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(x_val,y_val),
                    batch_size=batch_size,
                    shuffle=False
                )
                test_loader = DataLoader(
                    TensorDataset(x_test,y_test),
                    batch_size=batch_size,
                    shuffle=False
                )

                return train_loader,val_loader,test_loader

            else:
                train_loader = DataLoader(
                    TensorDataset(x_train,y_train),
                    batch_size=len(x_train),
                    shuffle=True
                )
                val_loader = DataLoader(
                    TensorDataset(x_val,y_val),
                    batch_size=len(x_val),
                    shuffle=False
                )
                test_loader = DataLoader(
                    TensorDataset(x_test,y_test),
                    batch_size=len(x_test),
                    shuffle=False
                )

                return train_loader,val_loader,test_loader
                 
        else:
            x_train = torch.tensor(X_train[:,:],dtype=torch.float32)
            x_val = torch.tensor(X_val[:,:],dtype=torch.float32)
            x_test = torch.tensor(X_test[:,:],dtype=torch.float32)

            if batch_size is not None:
                train_loader = DataLoader(
                        TensorDataset(x_train),
                        batch_size=batch_size,
                        shuffle=True
                    )
                val_loader = DataLoader(
                        TensorDataset(x_val),
                        batch_size=batch_size,
                        shuffle=False
                    )
                test_loader = DataLoader(
                        TensorDataset(x_test),
                        batch_size=batch_size,
                        shuffle=False
                    )

                return train_loader,val_loader,test_loader
            
            else:
                train_loader = DataLoader(
                        TensorDataset(x_train),
                        batch_size=len(x_train),
                        shuffle=True
                    )
                val_loader = DataLoader(
                        TensorDataset(x_val),
                        batch_size=len(x_val),
                        shuffle=False
                    )
                test_loader = DataLoader(
                        TensorDataset(x_test),
                        batch_size=len(x_test),
                        shuffle=False
                    )

                return train_loader,val_loader,test_loader
