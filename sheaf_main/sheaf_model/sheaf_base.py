from torch import nn

# custom imports
from .config import DEFAULT_DROPOUT, DEFAULT_INPUT_DROPOUT, DEFAULT_LEFT_WEIGHTS, DEFAULT_RIGHT_WEIGHTS
from utils.sheaf_utils import validate_edge_index

class SheafDiffusion(nn.Module):

    """
    Base class for sheaf diffusion implementation
    """

    def __init__(self, data, args):
        super().__init__()

        # Stalk dimension
        assert args["d"] > 0 # dimension of the stalk must be positive
        self.d = args["d"]

        self.num_nodes = data.num_nodes
        if getattr(data, "x", None) is None:
            raise ValueError("data.x is required to infer input_dim.")

        # Validate edge indices, check for self-loops and require bidirectionality.
        validate_edge_index(data.edge_index, num_nodes=self.num_nodes or None, require_unique_edges=True)
        self.edge_index = data.edge_index

        self.input_dim = data.num_node_features or data.x.size(-1)

        self.output_dim = args.get("output_dim")
        if self.output_dim is None:
            if getattr(data, "y", None) is None:
                raise ValueError("output_dim must be supplied when data.y is unavailable.")
            self.output_dim = int(data.y.max().item()) + 1

        self.hidden_channels = args.get("hidden_channels", 32)
        self.hidden_dim = self.hidden_channels * self.d

        self.layers = args.get("layers", 2)
        self.normalised = args.get("normalised", True)

        self.input_dropout = args.get('input_dropout', DEFAULT_INPUT_DROPOUT)
        self.dropout = args.get('dropout', DEFAULT_DROPOUT)

        self.left_weights = args.get('left_weights', DEFAULT_LEFT_WEIGHTS) # W1 in the paper, maybe initialize as identity
        self.right_weights = args.get('right_weights', DEFAULT_RIGHT_WEIGHTS) # W2 in the paper, maybe initialize as identity

        self.laplacian_builder = None

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
