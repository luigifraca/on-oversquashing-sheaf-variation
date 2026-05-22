# torch imports
import torch
from torch import nn
from torch_geometric.utils import degree
from torch_scatter import scatter_add

# custom imports
from utils.sheaf_utils import validate_edge_index
from utils.sheaf_utils import laplace as lap

# global config constants
from .config import DEFAULT_AUGMENTED_LAPLACIAN

class LaplacianBuilder(nn.Module):

    def __init__(self, size, edge_index, d, normalised=True, augmented=DEFAULT_AUGMENTED_LAPLACIAN):
        super().__init__()
        self.d = d
        self.size = size

        # edge validation
        validate_edge_index(edge_index, num_nodes=size, require_unique_edges=True)
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index

        self.normalised = normalised
        self.device = edge_index.device
        self.augmented = augmented # meaning, when normalising the Laplacian, use an augmented degree matrix that adds self-loops

        # Pair directed edge maps and store one canonical edge per undirected edge.
        self.left_right_idx, self.undirected_edge_index = lap.compute_left_right_map_index(edge_index)
        self.vertex_tril_idx = self.undirected_edge_index
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

class GeneralLaplacianBuilder(LaplacianBuilder):
    """Learns a multi-dimensional Sheaf Laplacian from data"""

    def __init__(self, size, edge_index, d, normalised=True, augmented=True):
        super().__init__(size, edge_index, d, normalised=normalised, augmented=augmented)

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = lap.compute_learnable_laplacian_indices(
            size,
            self.undirected_edge_index,
            self.d
        )

    def normalise(self, diag_maps, non_diag_maps, tril_row, tril_col):
        if self.normalised:
            # Normalise the entries if the normalised Laplacian is used.
            if self.training:
                # During training, we perturb the matrices to ensure they have different singular values.
                # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                eps = torch.FloatTensor(self.d).uniform_(-0.001, 0.001).to(device=self.device)
            else:
                eps = torch.zeros(self.d, device=self.device)

            to_be_inv_diag_maps = diag_maps + torch.diag(1. + eps).unsqueeze(0) if self.augmented else diag_maps
            d_sqrt_inv = lap.batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)
            assert torch.all(torch.isfinite(d_sqrt_inv))
            left_norm = d_sqrt_inv[tril_row]
            right_norm = d_sqrt_inv[tril_col]
            non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(min=-1, max=1)
            diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(min=-1, max=1)
            assert torch.all(torch.isfinite(non_diag_maps))
            assert torch.all(torch.isfinite(diag_maps))

        return diag_maps, non_diag_maps

    def forward(self, maps):
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.undirected_edge_index
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Compute transport maps.
        assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=self.size)

        # Normalise the transport maps.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # Merge diagonal and non-diagonal
        edge_index, weights = lap.mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        return (edge_index, weights), saved_tril_maps
