import torch
import torch_scatter
from torch.nn import Linear, Module
from torch_geometric.nn import MessagePassing, Aggregation, GAT
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

class SSAggregation(Aggregation):
    def __init__(self, ss_state_mat, ss_input_mat):
        super().__init__()
        self.ss_state_mat = ss_state_mat
        self.ss_input_mat = ss_input_mat

    def forward(self, x, index, **kwargs):
        print(f"hello aggregate forward inputs: {x.shape}, index: {index.shape}")
        # grouped = torch_scatter.scatter(x, index, dim=0) # this needs a reduction method

        # Sort by index so groups are contiguous
        sorted_index, perm = torch.sort(index)
        x_sorted = x[perm]
        # Find group boundaries (where index changes)
        boundaries = torch.where(sorted_index[1:] != sorted_index[:-1])[0] + 1
        boundaries = boundaries.cpu() # only the boundaries need to be on CPU
        groups = torch.tensor_split(x_sorted, boundaries)
        unique_nodes = sorted_index[boundaries - 1].unique(sorted=True)

        results = [self.ss_aggregate(g) for g in groups]

        print(f"before out results: {len(results)} results[0]: {results[0].shape}")
        out = torch.stack(results, dim=0)
        print(f"after out out: {out.shape}")
        return out

    def ss_aggregate(self, g):
        # print(f"hello aggregate olala g: {g.shape}")
        return torch.sum(g, dim=0)
        # TODO: self-messages with state matrix
        return torch.matmul(self.ss_input_mat, torch.sum(g, dim=0))

# state space message passing graph neural network
# x(t+1) = A * x(t) + B * u(t); dimx = dimu
class SSMPGNNLayer(MessagePassing):
    # we want to share the state and input matrices across layers, so they will be managed by a higher level structure
    def __init__(self, in_features, out_features, aggr):
        super().__init__(aggr=aggr)
        self.lin = Linear(in_features, out_features, bias=False)

    def forward(self, x, edge_index):
        print(f"FORWARD x: {x.shape} edge_index: {edge_index.shape}")
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        print(f"hello message x_j: {x_j.shape}")
        return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        print(f"hello update aggr_out: {aggr_out.shape}")
        return aggr_out

class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, mplayer_size, out_channels, node_feature_size):
        super().__init__()

        self.ss_state_mat = torch.rand((node_feature_size, node_feature_size)).cuda()
        self.ss_input_mat = torch.rand((node_feature_size, node_feature_size)).cuda()
        print(f"hello mygnn mat shapes: {self.ss_state_mat.shape}")

        self.aggr = SSAggregation(self.ss_state_mat, self.ss_input_mat)

        self.conv1 = SSMPGNNLayer(in_channels, mplayer_size, self.aggr)
        self.conv2 = SSMPGNNLayer(mplayer_size, mplayer_size, self.aggr)
        self.conv3 = SSMPGNNLayer(mplayer_size, mplayer_size, self.aggr)
        self.gat = GAT(mplayer_size, mplayer_size, 2, out_channels)
        # self.conv3 = SSMPGNNLayer(out_channels, out_channels, self.aggr)
        # self.conv4 = SSMPGNNLayer(out_channels, out_channels, self.aggr)
        # self.conv5 = SSMPGNNLayer(out_channels, out_channels, self.aggr)

    def forward(self, x, edge_index):
        # layer 1
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # layer 2
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.gat(x, )

        # # layer 3
        # x = self.conv3(x, edge_index)
        # x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # # layer 4
        # x = self.conv4(x, edge_index)
        # x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # # layer 5
        # x = self.conv5(x, edge_index)
        # x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return x