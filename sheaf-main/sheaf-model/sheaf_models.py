import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple
from abc import abstractmethod

from utils.sheaf_utils import laplace as lap
from config import DEFAULT_SHEAF_ACTIVATION

class SheafLearner(nn.Module):
    """Base model to learn sheaf from features and graph structure"""

    def __init__(self):
        super().__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()

class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act=DEFAULT_SHEAF_ACTIVATION):
        super().__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)

        # CAREFUL, THIS IS ROW-MAJOR ORDER, NOT COLUMN-MAJOR
        # [a11, a12, a21, a22] -> [[a11, a21], [a12, a22]]
        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])
