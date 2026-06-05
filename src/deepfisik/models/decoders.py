import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, GELU, LayerNorm
from torch.nn import Parameter
from torch.nn.functional import relu
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
import numpy as np

class nodeDecoder(torch.nn.Module):
    def __init__(self, embedding_size):
        super(nodeDecoder, self).__init__()

        self.node_d1 = Seq(Lin(embedding_size, int(embedding_size/2)),GELU(),LayerNorm(int(embedding_size/2)))
        self.node_d2 = Seq(Lin(int(embedding_size/2), int(embedding_size/4)),GELU(),LayerNorm(int(embedding_size/4)))
        self.node_d3 = Seq(Lin(int(embedding_size/4), 1),GELU(),LayerNorm(1))
        self.node_d4 = Lin(1,1)

    def forward(self, x):
        x = self.node_d1(x)
        x = self.node_d2(x)
        x = self.node_d3(x)
        x = self.node_d4(x)

        return x

class edgeDecoder(torch.nn.Module):
    def __init__(self, edge_dim, embedding_size):
        super(edgeDecoder, self).__init__()

        self.edge_embedding_size = embedding_size
        self.edge_dim_size = edge_dim
        if embedding_size > edge_dim:
            if int(embedding_size/4) >= edge_dim:
                self.edge_d1 = Seq(Lin(embedding_size, int(embedding_size/2)),GELU(),LayerNorm(int(embedding_size/2)))
                self.edge_d2 = Seq(Lin(int(embedding_size/2), int(embedding_size/4)),GELU(),LayerNorm(int(embedding_size/4)))
                self.edge_d3 = Seq(Lin(int(embedding_size/4), edge_dim),GELU(),LayerNorm(edge_dim))
                self.edge_d4 = Lin(edge_dim,edge_dim)
            elif int(embedding_size/4) < edge_dim :
                self.edge_d1 = Seq(Lin(embedding_size, edge_dim),GELU(),LayerNorm(edge_dim))

    def forward(self, x):

        if self.edge_embedding_size > self.edge_dim_size:
            if int(self.edge_embedding_size/4) >= self.edge_dim_size:
                x = self.edge_d1(x)
                x = self.edge_d2(x)
                x = self.edge_d3(x)
                x = self.edge_d4(x)
            elif int(self.edge_embedding_size/4) < self.edge_dim_size:
                x = self.edge_d1(x)
        elif self.edge_embedding_size == self.edge_dim_size:
            x = x

        return x

class globalDecoder(torch.nn.Module):
    def __init__(self):
        super(globalDecoder, self).__init__()

        self.u_d1 = Seq(Lin(48, 36),GELU(),LayerNorm(36))
        self.u_d2 = Seq(Lin(36,24),GELU(),LayerNorm(24))
        self.u_d3 = Seq(Lin(24, 1),GELU(),LayerNorm(1))
        self.u_d4 = Lin(1,1)

    def forward(self, x):
        x = self.u_d1(x)
        x = self.u_d2(x)
        x = self.u_d3(x)
        x = self.u_d4(x)

        return x
