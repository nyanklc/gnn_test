import torch
import torch_scatter
from torch.nn import Linear, Module
from torch_geometric.nn import MessagePassing, Aggregation, GAT
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from model import *

if __name__ == "__main__":
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
    ], dtype=torch.long)

    edge_index = to_undirected(edge_index)   # make bidirectional

    num_nodes = 5
    num_features = 8
    num_classes = 2

    x = torch.randn(num_nodes, num_features)
    y = torch.randint(0, num_classes, (num_nodes,))

    data = Data(x=x, edge_index=edge_index, y=y)

    print(f"DATA: {data}")
    print(f"x: {x}")
    print(f"edge_index: {edge_index}")
    print(f"y: {y}")

    # ------------------------------
    # Instantiate model
    # ------------------------------

    walk_embedding_size = 16 # embedding on walks
    att_embedding_size = 4 # embedding done by gat
    embedding_size = 128 # final embedding size
    att_heads = 1
    random_walk_length = 4 # including starting node
    neigh_aggr = "mean"
    walk_embedder = WalkEmbedder(random_walk_length=random_walk_length-1, embedding_size=walk_embedding_size)
    model = WalkingMessagesModel(
        neigh_aggr=neigh_aggr,
        att_in_channels=num_features,
        att_embedding_size=att_embedding_size,
        att_heads=att_heads,
        embedding_size=embedding_size,
        walk_embedder=walk_embedder,
    )

    _param_count = sum(p.numel() for p in model.parameters())
    print(f"model initialized, parameter count: {_param_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ------------------------------
    # Train
    # ------------------------------
    print("Starting training loop...")

    for epoch in range(20):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss = {loss.item():.4f}")