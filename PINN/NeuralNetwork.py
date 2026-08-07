import torch
from torch import nn

class MLP(nn.Module):
    """Predict concentration while enforcing the initial and centre conditions.

    Squared radius guarantees centre symmetry and the output transformation
    ``C0 + tau * N`` exactly recovers the initial concentration.
    """

    def __init__(self):
        super().__init__()

        self.input = nn.Linear(2,100)
        self.h1 = nn.Linear(100,100)
        self.h2 = nn.Linear(100,100)
        self.h3 = nn.Linear(100,100)
        self.out = nn.Linear(100,1)

        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight,gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(module.bias)

    def forward(self,x:torch.tensor,init_concentration):
        """Map ``(rho, tau)`` coordinates to normalized concentration."""
        rho = x[:,:1]
        tau = x[:,1:2]

        tx = torch.cat(
            (rho.square(),tau),                                                                 #square to hard impose first boundary condition (symmetry around the origin)
            dim=1
        )

        hidden = self.activation(self.input(tx))
        hidden = self.activation(self.h1(hidden))
        hidden = self.activation(self.h2(hidden))
        hidden = self.activation(self.h3(hidden))

        out = self.out(hidden)
        concentration = init_concentration + tau * out                                          #impose initial condition

        return  concentration

class FFN(nn.Module):
    """Learn the post-training correction to a frozen PINN field.

    The network uses ``rho``, ``tau``, and ``C_PINN`` and multiplies its output
    by time so that the learned correction vanishes initially.
    """

    def __init__(self):
        super().__init__()

        self.input = nn.Linear(3,32)
        self.h1 = nn.Linear(32,32)
        self.h2 = nn.Linear(32,32)
        self.h3 = nn.Linear(32,32)
        self.out = nn.Linear(32,1)                    #residual

        self.activation = nn.Tanh()

    def forward(self,X):
        """Return the concentration correction for residual-model inputs."""
        rho = X[:, 0:1]
        tau = X[:, 1:2]
        concentration_pinn = X[:, 2:3]

        features = torch.cat(
            (rho.square(), tau, concentration_pinn),
            dim=1
        )

        x = self.activation(self.input(features))
        h1 = self.activation(self.h1(x))
        h2 = self.activation(self.h2(h1))
        h3 = self.activation(self.h3(h2))

        out = self.out(h3)

        return tau * out


class HybridModel(nn.Module):
    """Combine a frozen PINN baseline with a learned residual correction.

    The wrapper exposes both components for diagnostics and their sum as the
    final Hybrid-PINN concentration prediction.
    """

    def __init__(self, pinn_model, residual_model):
        super().__init__()
        self.pinn_model = pinn_model
        self.residual_model = residual_model

    def forward_components(self, coordinates, initial_concentration):
        """Return baseline, residual correction, and corrected concentration."""
        concentration_pinn = self.pinn_model(
            coordinates,
            initial_concentration
        )
        residual_inputs = torch.cat(
            (coordinates, concentration_pinn),
            dim=1
        )
        residual_correction = self.residual_model(residual_inputs)
        concentration_hybrid = (
            concentration_pinn + residual_correction
        )

        return (
            concentration_pinn,
            residual_correction,
            concentration_hybrid
        )

    def forward(self, coordinates, initial_concentration):
        _, _, concentration_hybrid = self.forward_components(
            coordinates,
            initial_concentration
        )

        return concentration_hybrid
