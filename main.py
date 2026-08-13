import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

from model import WalkingMessagesModel, WalkEmbedder


def visualize_dataset(dataset):
    data = dataset[0]
    # Convert edge_index to NetworkX graph
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    # Use TSNE to get 2D coordinates for nodes
    tsne = TSNE(n_components=2, random_state=42)
    node_features_2d = tsne.fit_transform(data.x.numpy())

    # Create a dict for node positions
    pos = {i: node_features_2d[i] for i in range(data.num_nodes)}

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        node_color=data.y.numpy(),
        cmap='tab10',
        node_size=50,
        with_labels=False,
        edge_color='lightgray',
        alpha=0.7
    )
    plt.title("Cora citation graph with node labels (colored)")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

torch_geometric.seed_everything(2143466)

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)
visualize_dataset(dataset)

data = dataset[0].to(device)
print(data)
in_channels = dataset.num_features
out_channels = dataset.num_classes
print(f"loaded dataset")
print("========== Dataset Info ==========")
print(f"Dataset: {dataset.name}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features (input channels): {dataset.num_features}")
print(f"Number of classes (output channels): {dataset.num_classes}")
print("----------------------------------")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges // 2} (directed: {data.is_directed}")
print(f"Training nodes: {int(data.train_mask.sum())}")
print(f"Validation nodes: {int(data.val_mask.sum())}")
print(f"Test nodes: {int(data.test_mask.sum())}")
print("----------------------------------")
print(f"Node feature shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Labels shape: {data.y.shape}")
print(f"Device: {device}")
print("==================================\n")

######################################################
NR_EPOCHS = 100

ATT_IN_CHANNELS = in_channels # ?
ATT_OUT_CHANNELS = in_channels # ?
ATT_HEADS = 4
EMBEDDING_SIZE = 128
RW_LENGTH = 32
######################################################

model = WalkingMessagesModel(
  "mean",
  ATT_IN_CHANNELS,
  ATT_OUT_CHANNELS,
  ATT_HEADS,
  EMBEDDING_SIZE,   # output size
  WalkEmbedder(RW_LENGTH, EMBEDDING_SIZE)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(f"created model")

# TRAIN
model.train()
for epoch in range(NR_EPOCHS):
    print(f"main data.x: {data.x.shape}")
    print(f"main data.edge_index: {data.edge_index.shape}")
    optimizer.zero_grad()
    print(f"main before forward call data.x: {data.x.shape} data.edge_index: {data.edge_index.shape}")
    out = model(data.x, data.edge_index)
    print(f"main after forward call out: {out.shape} data.edge_index: {data.edge_index.shape}")
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    print(f"LOSS: {loss.item()}")
    input()
    loss.backward()
    optimizer.step()

# TEST
model.eval()
pred = model(data, data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
