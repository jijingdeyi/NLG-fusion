import torch.nn as nn

from utils.edge_detect import EdgeDetect


class edge(nn.Module):
    def __init__(self):
        super(edge, self).__init__()

        self.ed = EdgeDetect()

    def forward(self, x):
        # edge detect
        e = self.ed(x)
        x = x + e

        return x
