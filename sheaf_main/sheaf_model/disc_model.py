# torch imports
import torch
import torch_sparse
from torch import nn
import torch.nn.functional as F

# custom imports
from . import laplacian_builder as lb
from .sheaf_base import SheafDiffusion
from .sheaf_models import LocalConcatSheafLearner

class DiscreteGeneralSheafDiffusion(SheafDiffusion):
    """Learns a multi-dim Sheaf Laplacian from data, performs diffusion"""

    def __init__(self, data, args):
        super().__init__(data, args)

        self.lin_right_weights = nn.ModuleList([
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            for _ in range(self.layers)
        ]) if self.right_weights else nn.ModuleList()
        for layer in self.lin_right_weights:
            nn.init.orthogonal_(layer.weight)

        self.lin_left_weights = nn.ModuleList([
            nn.Linear(self.d, self.d, bias=False)
            for _ in range(self.layers)
        ]) if self.left_weights else nn.ModuleList()
        for layer in self.lin_left_weights:
            nn.init.eye_(layer.weight)

        self.sheaf_learners = nn.ModuleList([
            LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d, self.d))
            for _ in range(self.layers)
        ])

        self.laplacian_builder = self._build_laplacian_builder(self.num_nodes)

        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.d, 1)))
            for _ in range(self.layers)
        ])

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def _build_laplacian_builder(self, num_nodes):
        return lb.GeneralLaplacianBuilder(
            num_nodes,
            self.edge_index,
            d=self.d,
            normalised=self.normalised
        )

    def _ensure_laplacian_builder(self, num_nodes):
        if num_nodes <= 0:
            raise ValueError("Cannot run sheaf diffusion on an empty node feature matrix.")
        if self.num_nodes != num_nodes:
            self.num_nodes = num_nodes
            self.laplacian_builder = None
        if self.laplacian_builder is None or self.laplacian_builder.size != num_nodes:
            self.laplacian_builder = self._build_laplacian_builder(num_nodes)

    def left_right_linear(self, x, layer, num_nodes):

        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights[layer](x)
            x = x.reshape(-1, num_nodes * self.d).t()

        if self.right_weights:
            x = self.lin_right_weights[layer](x)

        return x

    def forward(self, x):
        num_nodes = x.size(0)
        self._ensure_laplacian_builder(num_nodes)
        self._reset_node_representations(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin12(x)
        x = x.view(num_nodes * self.d, -1)
        self._store_node_representation("encoded", x)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}

        for layer in range(self.layers):
            x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0, training=self.training)
            maps = self.sheaf_learners[layer](x_maps.reshape(num_nodes, -1), self.edge_index)
            L, trans_maps = self.laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            self._last_maps[layer] = maps
            self._last_trans_maps[layer] = trans_maps
            self._last_laplacian[layer] = L

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, layer, num_nodes)

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(num_nodes, 1)) * x0 - x
            x = x0
            self._store_node_representation(f"layer{layer}", x)

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(num_nodes, -1)
        self._store_node_representation("pre_logits", x)
        x = self.lin2(x)
        self._store_node_representation("logits", x)
        return F.log_softmax(x, dim=1)
