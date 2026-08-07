import torch
import torch.nn as nn

class FNN(nn.Module):
    """Predict a complete radial concentration-discrepancy profile.

    Average concentration, surface concentration, and current define the input;
    each output is the FVM--simplified correction at one radial node.
    """

    feature_names = (
        "average_concentration",
        "surface_concentration",
        "current",
    )

    def __init__(
            self, 
            input_dim: int = 3,
            output_dim: int = 101,
            hidden_dim: int = 32,
            ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = nn.Linear(input_dim, hidden_dim)
        self.h1 = nn.Linear(hidden_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (batch_size, input_dim)."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"received {features.shape[1]}."
            )

        x = self.activation(self.input(features))
        x = self.activation(self.h1(x))
        x = self.activation(self.h2(x))

        return self.out(x)


class ThermalFFN(nn.Module):
    """Approximate the lumped thermal derivative from state and current.

    The two inputs are normalized ``theta`` and current; the scalar output is
    normalized ``dtheta/dt`` and is integrated externally during rollout.
    """

    def __init__(
            self,
            device,
            input_dim=2,
            hidden_dim=32,
            output_dim=1
            ):
        super().__init__()

        self.device = device

        self.input = nn.Linear(input_dim,hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim,hidden_dim)
        self.output = nn.Linear(hidden_dim,output_dim)

        self.activation = nn.Tanh()

    def forward(self,X):
        """Return the normalized temperature derivative for each input state."""
        x = self.activation(self.input(X))
        h1 = self.activation(self.hidden1(x))
        out = self.output(h1)

        return out
