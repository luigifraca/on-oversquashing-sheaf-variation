import itertools
import math

import torch
import torch_sparse
from torch_geometric.utils import degree

from utils.sheaf_utils import validate_edge_index

def build_sheaf_laplacian(N: int, K: int, edge_index: torch.Tensor, maps: torch.Tensor):
    """
    Builds a sheaf laplacian given the edge_index and the restriction maps

    Args:
        N: The number of nodes in the graph
        K: The dimensionality of the Stalks
        edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
            edge_index[0, i] --> edge_index[1, i].
        maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
    Returns:
        (index, value): The sheaf Laplacian as a sparse matrix of size (N*K, N*K)
    """
    E = edge_index.size(1)
    index = []
    values = []

    for e in range(E):
        source = edge_index[0, e]
        target = edge_index[1, e]

        top_x = e * K
        # Generate the positions in the block matrix
        top_y = source * K
        for i, j in itertools.product(range(K), range(K)):
            index.append([top_x + i, top_y + j])
            values.append(-maps[e, 0, i, j])

        top_y = target * K
        for i, j in itertools.product(range(K), range(K)):
            index.append([top_x + i, top_y + j])
            values.append(maps[e, 1, i, j])

    index = torch.tensor(index, dtype=torch.long).T
    values = torch.tensor(values)

    index_t, values_t = torch_sparse.transpose(index, values, E * K, N * K)
    index, value = torch_sparse.spspmm(index_t, values_t, index, values, N * K, E * K, N * K, coalesced=True)
    return torch_sparse.coalesce(index, value, N * K, N * K)

def sym_matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    """
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: a batch of matrices
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.linalg.eigh(matrix)
    vals[vals > 0] = vals[vals > 0].pow(p)
    matrix_pow = vecs @ torch.diag(vals) @ vecs.T
    return matrix_pow


def build_norm_sheaf_laplacian(N: int, K: int, edge_index: torch.Tensor, maps: torch.Tensor, augmented: bool = True):
    """
    Builds a normalised sheaf laplacian given the edge_index and the restriction maps.

    Args:
        N: The number of nodes in the graph
        K: The dimensionality of the Stalks
        edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
            edge_index[0, i] --> edge_index[1, i].
        maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
        augmented: Use D* = D + I instead of D.
    Returns:
        (index, value): The normalised sheaf Laplacian as a sparse matrix of size (N*K, N*K)
    """
    index, values = build_sheaf_laplacian(N, K, edge_index, maps)
    block_diag_indices = []
    block_diag_values = []

    for i in range(N):
        low = i * K
        high = low + K

        mask1 = torch.logical_and(low <= index[0, :], index[0, :] < high)
        mask2 = torch.logical_and(low <= index[1, :], index[1, :] < high)
        mask = torch.logical_and(mask1, mask2)

        d_index = index[:, mask]
        d_values = values[mask]
        d_index = d_index - low

        Dv = torch.sparse_coo_tensor(d_index, d_values).to_dense()
        assert list(Dv.size()) == [K, K]
        if augmented:
            Dv = Dv + torch.eye(K, K)
        Dv_sqrt_inv = sym_matrix_pow(Dv, -0.5).to_sparse()

        block_diag_indices.append(Dv_sqrt_inv.indices() + low)
        block_diag_values.append(Dv_sqrt_inv.values())

    D_sqrt_inv_idx = torch.cat(block_diag_indices, dim=1)
    D_sqrt_val = torch.cat(block_diag_values, dim=0)

    tmp_idx, tmp_val = torch_sparse.spspmm(D_sqrt_inv_idx, D_sqrt_val, index, values, N * K, N * K, N * K,
                                           coalesced=True)
    index, value = torch_sparse.spspmm(tmp_idx, tmp_val, D_sqrt_inv_idx, D_sqrt_val, N * K, N * K, N * K,
                                       coalesced=True)
    return torch_sparse.coalesce(index, value, N * K, N * K)


def dirichlet_energy(L, f, size):
    """Returns the Dirichlet energy of the signal f under the sheaf Laplacian L."""
    right = torch_sparse.spmm(L[0], L[1], size, size, f)
    energy = f.t() @ right
    return energy.item()


def get_edge_index_dict(edge_index, undirected=True):
    """Computes a dictionary mapping the undirected edges in edge_index to an ID."""
    assert edge_index.size(1) % 2 == 0

    E = edge_index.size(1)
    edge_idx_dict = dict()
    next_id = 0

    for e in range(E):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if undirected:
            edge = tuple(sorted([source, target]))
        else:
            edge = tuple([source, target])

        # Generate or retrieve the edge index
        if edge not in edge_idx_dict:
            edge_idx_dict[edge] = next_id
            next_id += 1

    return edge_idx_dict


def compute_incidence_index(edge_index, d):
    """Computes the indices of a sheaf coboundary matrix from the edge_index of the graph."""
    assert edge_index.size(1) % 2 == 0

    edge_idx_dict = get_edge_index_dict(edge_index)
    index = []

    for edge in range(edge_index.size(1)):
        source = edge_index[0, edge].item()
        target = edge_index[1, edge].item()
        edge_key = tuple(sorted([source, target]))

        top_x = edge_idx_dict[edge_key] * d
        top_y = source * d
        for i, j in itertools.product(range(d), range(d)):
            index.append([top_x + i, top_y + j])

    incidence_index = torch.tensor(index, dtype=torch.long).T
    assert list(incidence_index.size()) == [2, edge_index.size(1) * (d ** 2)]
    return incidence_index


def compute_left_right_map_index(edge_index):
    """Computes indices for lower triangular matrix or full matrix"""
    validate_edge_index(edge_index)
    edge_to_idx = dict()
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        edge_to_idx[(source, target)] = e

    left_index, right_index = [], []
    row, col = [], []
    for e in range(edge_index.size(1)):
        source = edge_index[0, e].item()
        target = edge_index[1, e].item()
        if source < target:
            left_index.append(e)
            right_index.append(edge_to_idx[(target, source)])

            row.append(source)
            col.append(target)

    left_index = torch.tensor(left_index, dtype=torch.long, device=edge_index.device)
    right_index = torch.tensor(right_index, dtype=torch.long, device=edge_index.device)
    left_right_index = torch.vstack([left_index, right_index])

    row = torch.tensor(row, dtype=torch.long, device=edge_index.device)
    col = torch.tensor(col, dtype=torch.long, device=edge_index.device)
    new_edge_index = torch.vstack([row, col])

    assert len(left_index) == edge_index.size(1) // 2

    return left_right_index, new_edge_index

def compute_learnable_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])

    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1, 1).tile(1, 1, learned_d)
    col_template = torch.transpose(row_template, dim0=1, dim1=2)

    non_diag_row_indices = (row_template + total_d*row.reshape(-1, 1, 1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d*col.reshape(-1, 1, 1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d*diag.reshape(-1, 1, 1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d*diag.reshape(-1, 1, 1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices

def batched_sym_matrix_pow(matrices: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrices: A batch of matrices.
        p: Power.
        positive_definite: If positive definite
    Returns:
        Power of each matrix in the batch.
    """
    # vals, vecs = torch.linalg.eigh(matrices)
    # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
    vecs, vals, _ = torch.linalg.svd(matrices)
    good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
    vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
    matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
    return matrix_power

def mergesp(index1, value1, index2, value2):
    """Merges two sparse matrices with disjoint indices into one."""
    assert index1.dim() == 2 and index2.dim() == 2
    assert value1.dim() == 1 and value2.dim() == 1
    assert index1.size(1) == value1.numel()
    assert index2.size(1) == value2.numel()
    assert index1.size(0) == 2 and index2.size(0) == 2

    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val
