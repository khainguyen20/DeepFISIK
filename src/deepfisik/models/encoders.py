# Copyright (C) 2026, Jaqaman Lab - UTSouthwestern
#
# This file is part of DeepFISIK.
#
# DeepFISIK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepFISIK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepFISIK.  If not, see <http://www.gnu.org/licenses/>.

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, GELU, LayerNorm
from torch.nn import Parameter
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer
import numpy as np
from sys import exit

class nodeEncoder(torch.nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(nodeEncoder, self).__init__()

        self.embedding_dimension = embedding_size
        self.node_feature_size = feature_size

        if embedding_size > feature_size:
            if int(embedding_size/4) > feature_size:
                self.node_e1 = Seq(Lin(feature_size, int(embedding_size/4)),GELU(),LayerNorm(int(embedding_size/4)))
                self.node_e2 = Seq(Lin(int(embedding_size/4), int(embedding_size /2)),GELU(),LayerNorm(int(embedding_size /2)))
                self.node_e3 = Seq(Lin(int(embedding_size /2), embedding_size),GELU(),LayerNorm(embedding_size))
            elif int(embedding_size/4) <= feature_size:
                if int(embedding_size/2) > feature_size:
                    self.node_e1 = Seq(Lin(feature_size, int(embedding_size/2)),GELU(),LayerNorm(int(embedding_size/2)))
                    self.node_e2 = Seq(Lin(int(embedding_size/2), embedding_size),GELU(),LayerNorm(embedding_size))
                else:
                    self.node_e1 = Seq(Lin(feature_size, embedding_size),GELU(),LayerNorm(embedding_size))

        elif embedding_size < feature_size:
            print("Node embedding size has to be bigger than node feature size")
            exit()

    def forward(self, x):
        if self.embedding_dimension > self.node_feature_size:
            if int(self.embedding_dimension/4) > self.node_feature_size:
                x = self.node_e1(x)
                x = self.node_e2(x)
                x = self.node_e3(x)
            elif int(self.embedding_dimension/4) <= self.node_feature_size:
                if int(self.embedding_dimension/2) > self.node_feature_size:
                    x = self.node_e1(x)
                    x = self.node_e2(x)
                else:
                    x = self.node_e1(x)
        elif self.embedding_dimension == self.node_feature_size:
            x = x
        elif self.embedding_dimension < self.node_feature_size:
            print("Node embedding size has to be bigger than node feature size")
            exit()

        return x

class edgeEncoder(torch.nn.Module):
    def __init__(self, edge_dim, embedding_size):
        super(edgeEncoder, self).__init__()

        self.embedding_dimension = embedding_size
        self.edge_feature_size = edge_dim

        if embedding_size > edge_dim:
            if int(embedding_size/4) > edge_dim:
                self.edge_e1 = Seq(Lin(edge_dim, int(embedding_size/4)),GELU(),LayerNorm(int(embedding_size/4)))
                self.edge_e2 = Seq(Lin(int(embedding_size/4), int(embedding_size /2)),GELU(),LayerNorm(int(embedding_size /2)))
                self.edge_e3 = Seq(Lin(int(embedding_size /2), embedding_size),GELU(),LayerNorm(embedding_size))
            elif int(embedding_size/4) <= edge_dim:
                if int(embedding_size/2) > edge_dim:
                    self.edge_e1 = Seq(Lin(edge_dim, int(embedding_size/2)),GELU(),LayerNorm(int(embedding_size/2)))
                    self.edge_e2 = Seq(Lin(int(embedding_size/2), embedding_size),GELU(),LayerNorm(embedding_size))
                else:
                    self.edge_e1 = Seq(Lin(edge_dim, embedding_size),GELU(),LayerNorm(embedding_size))

        elif embedding_size < edge_dim:
            print("Edge embedding size has to be bigger than edge feature size")
            exit()

    def forward(self, x):

        if self.embedding_dimension > self.edge_feature_size:
            if int(self.embedding_dimension/4) > self.edge_feature_size:
                x = self.edge_e1(x)
                x = self.edge_e2(x)
                x = self.edge_e3(x)
            elif int(self.embedding_dimension/4) <= self.edge_feature_size:
                if int(self.embedding_dimension/2) > self.edge_feature_size:
                    x = self.edge_e1(x)
                    x = self.edge_e2(x)
                else:
                    x = self.edge_e1(x)
        elif self.embedding_dimension == self.edge_feature_size:
            x = x
        elif self.embedding_dimension < self.edge_feature_size:
            print("Edge embedding size has to be bigger than edge feature size")
            exit()

        return x
