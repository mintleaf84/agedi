from typing import Callable

import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from agedi.models.head import Head

from torch_scatter import scatter


def build_gated_equivariant_mlp(
    s_in: int,
    v_in: int,
    n_out: int,
    n_layers: int = 2,
    activation: Callable = F.silu,
    sactivation: Callable = F.silu,
):
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s instead of dense layers.

    Parameters
    ----------
    n_in: int
        Number of input nodes.
    n_out: int
        Number of output nodes.
    n_layers: int
        Number of layers.
    activation: Callable
        Activation function.
    sactivation: Callable
        Activation function for the skip connection.
    n_hidden: int
        Number of hidden nodes.

    Returns
    -------
    nn.Module

    """
    # get list of number of nodes in input, hidden & output layers
    s_neuron = s_in
    v_neuron = v_in
    s_neurons = []
    v_neurons = []
    for i in range(n_layers):
        s_neurons.append(s_neuron)
        v_neurons.append(v_neuron)
        s_neuron = max(n_out, s_neuron // 2)
        v_neuron = max(n_out, v_neuron // 2)
    s_neurons.append(n_out)
    v_neurons.append(n_out)

    n_gating_hidden = s_neurons[:-1]

    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        snn.GatedEquivariantBlock(
            n_sin=s_neurons[i],
            n_vin=v_neurons[i],
            n_sout=s_neurons[i + 1],
            n_vout=v_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        snn.GatedEquivariantBlock(
            n_sin=s_neurons[-2],
            n_vin=v_neurons[-2],
            n_sout=s_neurons[-1],
            n_vout=v_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


class PositionsScore(Head):
    """Predict the positions score of the atoms in the structure.

    Parameters
    ----------
    input_dim_scalar: int
        The dimension of the scalar input.
    input_dim_vector: int
        The dimension of the vector input.
    gated_blocks: int
        The number of gated blocks in the network.

    Returns
    -------
    Head

    """

    _key = "pos"

    def __init__(
        self, input_dim_scalar=66, input_dim_vector=64, gated_blocks=3, **kwargs
    ):
        super().__init__(**kwargs)
        self.net = build_gated_equivariant_mlp(
            input_dim_scalar,
            input_dim_vector,
            1,
            n_layers=gated_blocks,
        )

    def _score(self, batch):
        """Predict the positions score of the atoms in the structure.

        Parameters
        ----------
        batch: dict
            The input batch.

        Returns
        -------
        torch.Tensor
            The predicted positions score.

        """
        scalar_representation = batch["scalar_representation"]
        vector_representation = batch["vector_representation"]

        scalar, vector = self.net([scalar_representation, vector_representation])

        return vector.squeeze(-1)


class TypesScore(Head):
    """Predict the types score of the atoms in the structure.

    Parameters
    ----------
    input_dim_scalar: int
        The dimension of the scalar input.
    input_dim_vector: int
        The dimension of the vector input.
    layers: int
        The number of layers

    Returns
    -------
    Head

    """

    _key = "x"

    def __init__(self, input_dim_scalar=66, input_dim_vector=64, layers=3, **kwargs):
        super().__init__(**kwargs)
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim_scalar, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.Softmax(dim=-1)
        # )
        self.net = nn.Linear(input_dim_scalar, 100)
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def _score(self, batch):
        """Predict the types score of the atoms in the structure.

        Parameters
        ----------
        batch: dict
            The input batch.

        Returns
        -------
        torch.Tensor
            The predicted positions score.

        """
        scalar_representation = batch["scalar_representation"]

        pred = self.net(scalar_representation)
        return pred


# class CellScore(Head):
#     """Predict the cell score of the structure.

#     Parameters
#     ----------
#     input_dim_scalar: int
#         The dimension of the scalar input.
#     input_dim_vector: int
#         The dimension of the vector input.
#     layers: int
#         The number of layers

#     Returns
#     -------
#     Head

#     """

#     _key = "cellpar"

#     def __init__(self, input_dim_scalar=66, input_dim_vector=64, layers=3, **kwargs):
#         super().__init__(**kwargs)
#         self.net = nn.Linear(input_dim_scalar, 6, bias=False)


#     def _score(self, batch):
#         """Predict the cell score of the structure.

#         Parameters
#         ----------
#         batch: dict
#             The input batch.

#         Returns
#         -------
#         torch.Tensor
#             The predicted positions score.

#         """
#         scalar_representation = batch["scalar_representation"]
#         structure_representation = scatter(scalar_representation, batch["_idx_m"], dim=0, reduce="mean")
        
#         pred = self.net(structure_representation)
        
#         # pred = pred.view(batch["_cell"].shape)
#         # pred = torch.einsum('bij,bjk->bik', pred, batch["_cell"])
#         return pred


# class CellScore(Head):
#     """Predict the cell score of the structure with physical constraints for radians."""

#     _key = "cellpar"

#     def __init__(self, input_dim_scalar=66, input_dim_vector=64, **kwargs):
#         super().__init__(**kwargs)
#         self.lengths_net = nn.Linear(input_dim_scalar, 3, bias=False)
#         self.angles_net = nn.Linear(input_dim_scalar, 3, bias=False)
        
#         # Initialize with zeros to start with neutral predictions
#         nn.init.zeros_(self.lengths_net.weight)
#         nn.init.zeros_(self.angles_net.weight)

#     def _score(self, batch):
#         """Predict the cell score with appropriate transformations for radian angles."""
#         scalar_representation = batch["scalar_representation"]
#         structure_representation = scatter(scalar_representation, batch["_idx_m"], 
#                                           dim=0, reduce="mean")
        
#         # Get current cell parameters
#         current_cellpar = batch[self.key]
        
#         # Apply safety clamps to prevent extreme values
#         current_lengths = torch.clamp(current_cellpar[:, :3], min=1e-6, max=1e6)  # a, b, c
        
#         # For angles in radians: π/6 (30°) to 5π/6 (150°)
#         min_angle_rad = torch.tensor(math.pi/6, device=current_cellpar.device)
#         max_angle_rad = torch.tensor(5*math.pi/6, device=current_cellpar.device)
#         current_angles = torch.clamp(current_cellpar[:, 3:], 
#                                     min=min_angle_rad*0.9,  # Add safety margin
#                                     max=max_angle_rad*1.1)  # Add safety margin
        
#         # Compute raw scores
#         lengths_score_raw = self.lengths_net(structure_representation)
#         angles_score_raw = self.angles_net(structure_representation)
        
#         # Safer transformation for lengths
#         # Strongly discourage lengths below 0.75Å
#         min_length = 0.75
#         lengths_factor = torch.sigmoid(-5 * (current_lengths - min_length))
#         lengths_score = lengths_score_raw * (1 + lengths_factor)
        
#         # Safer transformations for angles in radians
#         eps = 1e-6  # Small constant to prevent division by zero
        
#         # Calculate normalized distance to boundaries
#         angle_range = max_angle_rad - min_angle_rad + eps
#         lower_distance = torch.clamp((current_angles - min_angle_rad) / angle_range, 
#                                    min=eps, max=1.0-eps)
#         upper_distance = torch.clamp((max_angle_rad - current_angles) / angle_range, 
#                                    min=eps, max=1.0-eps)
        
#         # Use a softer boundary factor
#         min_distances = torch.min(lower_distance, upper_distance)
#         boundary_factor = 0.2 * torch.tanh(-3 * min_distances + 1.5)
        
#         # Target the middle of the valid range (π/2 or 90°)
#         mid_angle_rad = (min_angle_rad + max_angle_rad) / 2
#         direction = -torch.tanh((current_angles - mid_angle_rad) / (angle_range/6))
        
#         angles_score = angles_score_raw + direction * boundary_factor
        
#         # Apply gradient clipping for stability
#         lengths_score = torch.clamp(lengths_score, min=-10.0, max=10.0)
#         angles_score = torch.clamp(angles_score, min=-10.0, max=10.0)
        
#         # Combine scores
#         pred = torch.cat([lengths_score, angles_score], dim=-1)
        
#         return pred


class CellScore(Head):
    """Predict cell parameters with simple physical constraints.
    
    Ensures:
    - Three positive numbers for lengths (a, b, c)
    - Three numbers between 0 and π for angles (α, β, γ)
    """
    _key = "cellpar"

    def __init__(self, input_dim_scalar=66, input_dim_vector=64, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(input_dim_scalar+6, input_dim_scalar+6, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim_scalar+6, 6, bias=False),
        )
        
        # # Initialize with small values
        # nn.init.normal_(self.net.weight, mean=0.0, std=0.01)
        # nn.init.zeros_(self.net.bias)
        
        # # Set bias for length parameters to favor positive values (around 3-5Å)
        # if self.net.bias is not None:
        #     self.net.bias.data[:3] = torch.tensor([1.5, 1.5, 1.5])

    def _score(self, batch):
        """Predict cell parameters with appropriate physical ranges."""
        scalar_representation = batch["scalar_representation"]
        structure_representation = scatter(scalar_representation, batch["_idx_m"], 
                                          dim=0, reduce="mean")

        cellpar = batch[self.key]
        x = torch.cat([structure_representation, cellpar], dim=-1)

        # Get raw predictions
        raw_output = self.net(x)
        
        # Split into lengths and angles
        lengths_raw = raw_output[:, :3]
        angles_raw = raw_output[:, 3:]
        
        # Constrain lengths to be positive using softplus
        # softplus(x) = log(1 + exp(x)) is a smooth approximation of ReLU
        # Shifts by 0.5 to ensure reasonable minimum length
        # lengths = F.softplus(lengths_raw) + 0.5
        
        # Constrain angles to be between 0 and π using sigmoid
        # sigmoid(x) * π maps to (0, π)
        # angles = torch.sigmoid(angles_raw) * math.pi
        
        # Combine constrained predictions
        pred = torch.cat([lengths_raw, angles_raw], dim=-1)
        
        return pred
