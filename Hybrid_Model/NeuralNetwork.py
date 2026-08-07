import torch
import torch.nn as nn

class FNN(nn.Module):
    """Predict the radial concentration-discrepancy profile.

    The default input order is::

        [C_average, C_surface, current]

    Each output column corresponds to one radial node::

        delta_C[:, i] = C_reference[:, i] - C_simplified[:, i]

    The radial coordinates are metadata used to order the output columns; they
    are not network inputs. The corrected profile must be assembled outside the
    network as ``C_hybrid = C_simplified + delta_C``.
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
        x = self.activation(self.input(X))
        h1 = self.activation(self.hidden1(x))
        out = self.output(h1)

        return out