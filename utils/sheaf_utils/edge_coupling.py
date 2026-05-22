import torch

def validate_edge_index(edge_index, num_nodes=None, require_bidirectional=True):
    """Validate the edge_index invariants expected by sheaf Laplacian builders."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Expected edge_index with shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() == 0:
        return edge_index
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if edge_index.dtype not in integer_dtypes:
        raise ValueError(f"edge_index must contain integer indices, got dtype {edge_index.dtype}")
    if torch.any(edge_index < 0):
        raise ValueError("edge_index contains negative node indices.")
    if num_nodes is not None and torch.any(edge_index >= int(num_nodes)):
        raise ValueError(f"edge_index contains node indices outside [0, {int(num_nodes) - 1}].")
    if torch.any(edge_index[0] == edge_index[1]):
        raise ValueError("edge_index contains self-loops, which are not valid sheaf edges.")

    if require_bidirectional:
        edge_pairs = {(int(u), int(v)) for u, v in edge_index.detach().cpu().t().tolist()}
        missing_reverse = [(u, v) for u, v in edge_pairs if (v, u) not in edge_pairs]
        if missing_reverse:
            raise ValueError(f"edge_index is missing reverse edges, e.g. {missing_reverse[:5]}.")

    return edge_index


def undirected_edge_set(edge_index):
    validate_edge_index(edge_index, require_bidirectional=False)
    return {
        tuple(sorted((int(u), int(v))))
        for u, v in edge_index.detach().cpu().t().tolist()
        if int(u) != int(v)
    }


def _lexsort_keys(rows, cols, width=None):
    if rows.numel() == 0:
        return rows
    if width is None:
        width = int(torch.maximum(rows.max(), cols.max()).item()) + 1
    return rows.long() * int(width) + cols.long()


def sort_edge_index_with_values(edge_index, values):
    """Sort directed edges and carry any row-aligned values with them."""
    validate_edge_index(edge_index, require_bidirectional=False)
    if edge_index.size(1) != values.size(0):
        raise ValueError(
            f"edge_index has {edge_index.size(1)} edges but values has {values.size(0)} rows."
        )
    keys = _lexsort_keys(edge_index[0], edge_index[1])
    perm = torch.argsort(keys, stable=True)
    return edge_index[:, perm], values.index_select(0, perm)


def sort_sparse_entries(indices, values, width=None):
    """Sort sparse COO entries by row then column while preserving index/value coupling."""
    if indices.dim() != 2 or indices.size(0) != 2:
        raise ValueError(f"Expected sparse indices with shape [2, N], got {tuple(indices.shape)}")
    if values.dim() != 1:
        raise ValueError(f"Expected sparse values with shape [N], got {tuple(values.shape)}")
    if indices.size(1) != values.numel():
        raise ValueError(
            f"Sparse indices contain {indices.size(1)} entries but values contain {values.numel()}."
        )
    keys = _lexsort_keys(indices[0], indices[1], width=width)
    perm = torch.argsort(keys, stable=True)
    return indices[:, perm], values.index_select(0, perm)


def laplacian_matrix_to_edge_weights(
    laplacian,
    num_nodes,
    stalk_dim=1,
    reducer="mean",
    edge_index=None,
    strict_edge_index=True,
    return_entries=False,
):
    """
    Collapse off-diagonal Laplacian COO entries into undirected edge weights.

    The returned dictionary is sorted by canonical edge key. If edge_index is
    supplied, the extracted Laplacian edges are checked against it so downstream
    curvature rows cannot drift away from the graph topology.
    """
    if isinstance(laplacian, tuple):
        indices, values = laplacian
        rows = indices[0].detach().cpu().long()
        cols = indices[1].detach().cpu().long()
        vals = values.detach().cpu().abs().float()
    else:
        laplacian = laplacian.detach().cpu()
        if laplacian.dim() != 2 or laplacian.size(0) != 3:
            raise ValueError(f"Expected Laplacian matrix with shape [3, N], got {tuple(laplacian.shape)}")
        rows = laplacian[0].long()
        cols = laplacian[1].long()
        vals = laplacian[2].abs().float()

    num_nodes = int(num_nodes)
    if stalk_dim is None:
        max_index = int(torch.cat([rows, cols]).max().item()) if rows.numel() else -1
        stalk_dim = max(1, int((max_index + num_nodes) // num_nodes))
    stalk_dim = int(stalk_dim)

    buckets = {}
    entry_buckets = {}
    for row, col, value in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        u, v = int(row // stalk_dim), int(col // stalk_dim)
        if u == v or u >= num_nodes or v >= num_nodes:
            continue
        edge = tuple(sorted((u, v)))
        buckets.setdefault(edge, []).append(float(value))
        entry_buckets.setdefault(edge, []).append((int(row), int(col), float(value)))

    if edge_index is not None and strict_edge_index:
        expected = undirected_edge_set(edge_index)
        observed = set(buckets)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                "Laplacian edge set does not match edge_index: "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )

    if reducer == "mean":
        edge_weights = {edge: float(torch.tensor(values).mean().item()) for edge, values in buckets.items()}
    elif reducer == "max":
        edge_weights = {edge: max(values) for edge, values in buckets.items()}
    elif reducer == "sum":
        edge_weights = {edge: float(sum(values)) for edge, values in buckets.items()}
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")

    edge_weights = dict(sorted(edge_weights.items()))
    if not return_entries:
        return edge_weights
    return edge_weights, {edge: sorted(entry_buckets[edge]) for edge in edge_weights}
