# Standard Torch imports
import torch
from torch import nn

# custom imports
from config import DEFAULT_SHEAF_ACTIVATION, DEFAULT_INPUT_DROPOUT, DEFAULT_DROPOUT, DEFAULT_RIGHT_WEIGHTS, DEFAULT_LEFT_WEIGHTS
from utils.sheaf_utils import validate_edge_index

class SheafDiffusion(nn.Module):

    """
    Base class for sheaf diffusion implementation
    """

    def __init__(self, edge_index, args):
        super().__init__()

        # Stalk dimension
        assert args["d"] > 0 # dimension of the stalk must be positive
        self.d = args["d"]

        # Validate edge indices, check for self-loops and require bidirectionality
        validate_edge_index(edge_index)
        self.edge_index = edge_index

        self.hidden_channels = args['hidden_channels']
        self.hidden_dim = self.hidden_channels * self.d
        self.device = args['device']
        self.graph_size = args['graph_size'] # maybe not needed?
        self.layers = args['layers']
        self.normalised = args['normalised']

        self.input_dropout = DEFAULT_INPUT_DROPOUT
        self.dropout = DEFAULT_DROPOUT

        self.left_weights = DEFAULT_LEFT_WEIGHTS # W1 in the paper, maybe initialize as identity
        self.right_weights = DEFAULT_RIGHT_WEIGHTS # W2 in the paper, maybe initialize as identity

        # for Graph Transfer tasks, I/O dim = 5
        # for Signal Propagation dim is irrelevant, yet features vector norm must be 1
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']

        self.orth_trans = args['orth']

        self.sheaf_act = DEFAULT_SHEAF_ACTIVATION # use tanh or leakyReLU/ReLU/ELU

        self.laplacian_builder = None

    def update_edge_index(self, edge_index):
        validate_edge_index(edge_index, num_nodes=self.graph_size)
        self.edge_index = edge_index
        self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    # Following methods are for storing intermediate node representations for analysis/visualization
    # Not used in forward pass
    def _reset_node_representations(self, x=None):
        self._last_node_representations = {}
        if x is not None:
            self._store_node_representation("input", x)

    def _store_node_representation(self, name, x):
        x_detached = x.detach()
        if x_detached.dim() == 1:
            x_detached = x_detached.unsqueeze(-1)
        if x_detached.size(0) == self.graph_size * self.d:
            x_detached = x_detached.reshape(self.graph_size, -1)
        elif x_detached.size(0) == self.graph_size:
            x_detached = x_detached.reshape(self.graph_size, -1)
        self._last_node_representations[name] = x_detached.cpu()
