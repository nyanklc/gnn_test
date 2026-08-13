import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.nn import MessagePassing, GATConv
from torch import Tensor
from util import anonymous_walk


# in order to be able to use attention easily, just getting messages
class GATMessages(GATConv):
    def aggregate(self, inputs, index):
        return inputs

# f
class WalkEmbedder(nn.Module):
    def __init__(self, random_walk_length: int, embedding_size: int):
        super().__init__()

        self.len_rw = random_walk_length
        self.len_out = embedding_size

        self.weight = nn.Parameter(
            torch.empty(
                2 * (self.len_rw + 1), self.len_out
            )  # the rw sequence includes the initial node also (+1)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, walk):
        anon = anonymous_walk(walk)
        print(f"WALKEMBEDDER: walk: {walk}, anonymous: {anon}")
        print(f"WALKEMBEDDER: walk: {walk.shape}, anonymous: {anon.shape}")
        print(f"WALKEMBEDDER: cat: {torch.cat((walk, anon), dim=1)}, {torch.cat((walk, anon), dim=1).shape}")
        print(f"WALKEMBEDDER: self.weight: {self.weight.shape}")
        print(f"WALKEMBEDDER: walk.dtype: {walk.dtype}")
        print(f"WALKEMBEDDER: anon.dtype: {anon.dtype}")
        out = torch.matmul(torch.cat((walk, anon), dim=1).float(), self.weight)
        return out


class WalkingMessagesLayer(MessagePassing):
    def __init__(
        self,
        neigh_aggr,  # mean
        att_in_channels,
        att_embedding_size,
        att_heads,
        embedding_size,  # output size
        walk_embedder,
    ):
        super().__init__(aggr=neigh_aggr)

        # handles the attention based message passing on the random walks
        # TODO: self loops?
        self.gat = GATMessages(
            att_in_channels, att_embedding_size, heads=att_heads, add_self_loops=False
        )
        self.embedding_size = embedding_size
        self.walk_embedder = walk_embedder
        self.weight_matrix = nn.Parameter(
            torch.empty(att_embedding_size + walk_embedder.len_out, self.embedding_size)
        )
        nn.init.xavier_uniform_(self.weight_matrix)

    # TODO should we sample random walks offline?
    # x_j is source
    # the parameters are automatically inferred by torch, i don't really like it but whatever
    def message(self, x_j: Tensor, edge_index_j, edge_index_i, edge_index, x) -> Tensor:
        print(
            f"message called x_j: {x_j}, edge_index_j: {edge_index_j}, edge_index_i: {edge_index_i}, x: {x}"
        )

        # take a random walk from x_j
        row, col = edge_index
        print(f"row: {row}, col: {col}")
        len_rw = self.walk_embedder.len_rw
        # one walk per node
        # walk_edge_indices = torch.zeros((x_j.shape[0]), dtype=torch.int64) # nr of nodes
        # for i in range(x_j.shape[0]): walk_edge_indices[i] = edge_index_i[i]
        # print(f"WALK EDGE INDICES: {walk_edge_indices}")
        walk = random_walk(row, col, row, len_rw)
        print(f"WALK: {walk}, WALK SHAPE: {walk.shape}")

        # apply GAT on the rw samples (artificial edge indices)
        edges_src = walk.flatten()
        edges_dst = edge_index_i.repeat_interleave(len_rw+1)
        print(f"GAT: edges_src: {edges_src}, edges_dst: {edges_dst}")
        rw_edge_index = torch.stack(
            [edges_src, edges_dst], dim=0
        )  # TODO: correct direction?
        print(f"GAT: rw_edge_index: {rw_edge_index}, rw_edge_index.shape: {rw_edge_index.shape}")
        out_gat = self.gat(x, rw_edge_index)
        # TODO: the gat returns the embeddings, but we want to return message per edge at the end
        # we can artificially duplicate these embeddings
        print(f"GAT: out_gat shape: {out_gat.shape}")
        print(f"GAT: out_gat: {out_gat}")

        # get rw/aw embeddings
        print(f"walk shape: {walk.shape}")
        out_we = self.walk_embedder(walk)

        # apply update
        out = torch.matmul(torch.cat((out_gat, out_we), dim=1), self.weight_matrix)
        print(f"final out shape: {out.shape}")
        return F.sigmoid(out)

    def forward(self, x, edge_index):
        print(f"forward x: {x.shape} edge_index: {edge_index.shape}")
        return self.propagate(edge_index, x=x)


class WalkingMessagesModel(nn.Module):
    def __init__(
        self,
        neigh_aggr,  # "mean"
        att_in_channels,
        att_embedding_size,
        att_heads,
        embedding_size,  # output size
        walk_embedder,
    ):
        super().__init__()

        self.conv1 = WalkingMessagesLayer(
            neigh_aggr,
            att_in_channels,
            att_embedding_size,
            att_heads,
            embedding_size,
            walk_embedder,
        )
        self.conv2 = WalkingMessagesLayer(
            neigh_aggr,
            att_in_channels,
            att_embedding_size,
            att_heads,
            embedding_size,
            walk_embedder,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
