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

    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for _ in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for _ in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        for _ in range(self.layers):

            self.sheaf_learners.append(LocalConcatSheafLearner(
                self.hidden_dim,
                out_shape=(self.d, self.d))
            )

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            normalised=self.normalised
        )

        self.epsilons = nn.ParameterList()
        for _ in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):

        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        self._reset_node_representations(x)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin12(x)
        x = x.view(self.graph_size * self.d, -1)
        self._store_node_representation("encoded", x)

        x0, L = x, None
        self._last_maps = {}
        self._last_trans_maps = {}
        self._last_laplacian = {}

        for layer in range(self.layers):
            x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0, training=self.training)
            maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
            L, trans_maps = self.laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            self._last_maps[layer] = maps
            self._last_trans_maps[layer] = trans_maps
            self._last_laplacian[layer] = L

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0
            self._store_node_representation(f"layer{layer}", x)

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        self._store_node_representation("pre_logits", x)
        x = self.lin2(x)
        self._store_node_representation("logits", x)
        return F.log_softmax(x, dim=1)
