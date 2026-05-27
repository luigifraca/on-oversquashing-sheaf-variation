import torch
from torch import nn

# custom imports
from .config import DEFAULT_DROPOUT, DEFAULT_INPUT_DROPOUT, DEFAULT_LEFT_WEIGHTS, DEFAULT_RIGHT_WEIGHTS
from utils.sheaf_utils import validate_edge_index

class SheafDiffusion(nn.Module):

    """
    Base class for sheaf diffusion implementation
    """

    def __init__(self, args):
        super().__init__()

        # Stalk dimension
        self.d = self._arg(args, "d")
        assert self.d > 0 # dimension of the stalk must be positive

        self.input_dim = self._arg(args, "input_dim")
        if self.input_dim is None:
            raise ValueError("input_dim must be supplied for graph-dynamic sheaf models.")

        self.output_dim = self._arg(args, "output_dim")
        if self.output_dim is None:
            raise ValueError("output_dim must be supplied for graph-dynamic sheaf models.")

        self.hidden_channels = self._arg(args, "hidden_channels", 32)
        self.hidden_dim = self.hidden_channels * self.d

        self.layers = self._arg(args, "layers", 2)
        self.normalised = self._arg(args, "normalised", True)

        self.input_dropout = self._arg(args, 'input_dropout', DEFAULT_INPUT_DROPOUT)
        self.dropout = self._arg(args, 'dropout', DEFAULT_DROPOUT)

        self.left_weights = self._arg(args, 'left_weights', DEFAULT_LEFT_WEIGHTS) # W1 in the paper, maybe initialize as identity
        self.right_weights = self._arg(args, 'right_weights', DEFAULT_RIGHT_WEIGHTS) # W2 in the paper, maybe initialize as identity

        self.num_nodes = None
        self.edge_index = None
        self.laplacian_builder = None

    @staticmethod
    def _arg(args, name, default=None):
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    @staticmethod
    def _prepare_edge_index(edge_index, num_nodes):
        edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
        validate_edge_index(edge_index, num_nodes=num_nodes, require_unique_edges=True)
        return edge_index

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
        if self.num_nodes and x_detached.size(0) in (self.num_nodes, self.num_nodes * self.d):
            x_detached = x_detached.reshape(self.num_nodes, -1)
        self._last_node_representations[name] = x_detached.cpu()
