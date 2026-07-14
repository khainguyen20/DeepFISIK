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
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList, GELU
from torch_geometric.nn import TransformerConv, TopKPooling, GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import QuantileAggregation
from deepfisik.models.encoders import *
from deepfisik.models.decoders import *
torch.manual_seed(42)
from sys import exit

class GNNTrimer(torch.nn.Module):
    def __init__(self, model_params):
        super(GNNTrimer, self).__init__()
        node_embedding_size = model_params["node_model_embedding_size"][0] # size of node embeddings
        edge_model_embedding_size = model_params["edge_model_embedding_size"][0] # size of edge embeddings
        n_heads = model_params["model_attention_heads"][0] # number of heads
        attention_size = model_params["model_attention_head_dimension"][0] # size of attention head
        self.n_layers = model_params["model_layers"][0] # number of layers
        attention_dropout_rate = model_params["model_attention_dropout_rate"][0] # dropout rate for attention mechanism
        top_k_ratio = model_params["model_top_k_ratio"][0]
        self.top_k_every_n = model_params["model_top_k_every_n"][0]
        edge_dim = model_params["model_edge_dim"][0] # number of edge features
        feature_size = model_params["model_feature_size"][0] # number of dimensions of node features
        batch_size = model_params["model_batch_size"][0] # batch
        laplacian_size = model_params["model_laplacian"][0] # number of eigenvectors

        self.n_heads = n_heads # number of attention heads
        self.layer_dropout_rate = model_params["model_layer_dropout_rate"][0]

        

        if node_embedding_size > attention_size:
            print("Attention head dimension must be bigger than node embedding dimension")
            exit()

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        self.node_encode = nodeEncoder(feature_size,node_embedding_size)
        self.edge_encode = edgeEncoder(edge_dim,edge_model_embedding_size)
        self.conv1 = TransformerConv(int(node_embedding_size),
                                    int(attention_size/n_heads),
                                    heads=n_heads,
                                    dropout=attention_dropout_rate,
                                    edge_dim=edge_model_embedding_size
                                    )

        self.transf1 = Linear(attention_size, attention_size)
        self.bn1 = BatchNorm1d(attention_size)

        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(int(attention_size),
                                                    int(attention_size/n_heads),
                                                    heads=n_heads,
                                                    dropout=attention_dropout_rate,
                                                    edge_dim=edge_model_embedding_size
                                                    ))

            self.transf_layers.append(Linear(attention_size, attention_size))
            self.bn_layers.append(BatchNorm1d(attention_size))

        self.edge_decode = edgeDecoder(edge_dim, edge_model_embedding_size)

        self.linear0 = Linear(attention_size*9, node_embedding_size*4)
        self.linear4_2 = Linear(node_embedding_size*4, node_embedding_size*2)
        self.linear1 = Linear(node_embedding_size*2, node_embedding_size)
        self.linear2 = Linear(node_embedding_size, int(node_embedding_size/2))
        self.linear3 = Linear(int(node_embedding_size/2), int(node_embedding_size/4))
        self.DCOut = Linear(int(node_embedding_size/4), 1)
        self.AP2Out = Linear(int(node_embedding_size/4), 1)
        self.DR2Out = Linear(int(node_embedding_size/4), 1)
        self.AP3Out = Linear(int(node_embedding_size/4), 1)
        self.DR3Out = Linear(int(node_embedding_size/4), 1)
        self.RDOut = Linear(int(node_embedding_size/4), 1)
        self.LFOut = Linear(int(node_embedding_size/4), 1)

    def forward(self, x, edge_index,edge_attr, batch_index, device):
        x = self.node_encode(x)
        edge_attr = self.edge_encode(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.gelu(self.transf1(x))
        x = self.bn1(x)

        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.nn.functional.gelu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        q = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        quantiles = QuantileAggregation(q).to(device)
        global_representation.append(quantiles(x,batch_index))
        u = sum(global_representation)

        u = torch.nn.functional.gelu(self.linear0(u))
        u = torch.nn.functional.gelu(self.linear4_2(u))
        u = torch.nn.functional.gelu(self.linear1(u))
        u = torch.nn.functional.gelu(self.linear2(u))
        u = torch.nn.functional.gelu(self.linear3(u))
        DC = self.DCOut(u)
        AP2 = self.AP2Out(u)
        DR2 = self.DR2Out(u)
        AP3 = self.AP3Out(u)
        DR3 = self.DR3Out(u)
        RD = self.RDOut(u)
        LF = self.LFOut(u)
        edge_attr = self.edge_decode(edge_attr)
        return x, edge_attr, u,DC,AP2,AP3, DR2,DR3,RD,LF #gelu1, gelu2#, attention_coeffs
