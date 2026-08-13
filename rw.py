import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Utilities ----------
def row_normalize_adj(adj):
    """adj: (N,N) torch tensor (float) adjacency (can be sparse dense).
       Returns row-normalized transition matrix P where rows sum to 1.
    """
    row_sum = adj.sum(dim=1, keepdim=True)  # (N,1)
    # avoid div by zero
    row_sum[row_sum == 0] = 1.0
    P = adj / row_sum
    return P

# ---------- RandomWalkLayer ----------
class RandomWalkLayer(nn.Module):
    """Single GNN layer that aggregates node features using random walks.
       - P: (N,N) transition probabilities (row-normalized)
       - num_walks: how many walks to sample per source node
       - walk_length: how many steps per walk (the aggregator includes visited nodes)
       - aggregator: 'mean' | 'sum' | 'rnn' (rnn uses GRU over features along the walk)
    """
    def __init__(self, in_dim, out_dim, P, num_walks=8, walk_length=4, aggregator='mean', teleport_prob=0.0, device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.aggregator = aggregator
        self.teleport_prob = teleport_prob
        self.device = device

        # shared MLP after aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # optional RNN aggregator
        if aggregator == 'rnn':
            self.gru = nn.GRU(input_size=in_dim, hidden_size=in_dim, batch_first=True)

        # P is stored as a tensor on device (row-stochastic)
        # Expect P shape (N,N)
        self.register_buffer('P', P.to(device))

    def sample_walks(self, source_nodes):
        """
        Sample random walks starting from source_nodes.
        Args:
            source_nodes: LongTensor (B,) indices of nodes we want representations for
        Returns:
            walks: LongTensor (B, num_walks, walk_length+1) — node indices visited (including start node)
        """
        B = source_nodes.size(0)
        N = self.P.size(0)
        # We'll produce indices in shape (B, num_walks, walk_length+1)
        walks = torch.empty((B, self.num_walks, self.walk_length + 1), dtype=torch.long, device=self.device)
        # set start nodes
        walks[:, :, 0] = source_nodes.view(B, 1).expand(B, self.num_walks)

        # current nodes (B * num_walks) flattened
        curr = walks[:, :, 0].reshape(-1)  # (B * num_walks,)

        for t in range(1, self.walk_length + 1):
            # sample next node for each current node according to P[curr]
            # P[curr] has shape (B*num_walks, N)
            probs = self.P[curr]  # (B*num_walks, N)
            # incorporate teleport (teleport to original source) if desired:
            if self.teleport_prob > 0.0:
                # teleport to the very first node in that walk (store start)
                starts = walks[:, :, 0].reshape(-1)  # (B*num_walks,)
                # Build a mixed distribution: (1 - alpha) * probs + alpha * one-hot(starts)
                # Create teleport one-hot:
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, starts.unsqueeze(1), 1.0)
                probs = (1.0 - self.teleport_prob) * probs + self.teleport_prob * one_hot
            # To avoid numerical issues, renormalize rows:
            row_sum = probs.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1.0
            probs = probs / row_sum

            # sample with multinomial
            # torch.multinomial needs 2D probs and num_samples
            next_nodes = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B*num_walks,)
            walks[:, :, t] = next_nodes.view(B, self.num_walks)
            curr = next_nodes

        return walks  # (B, num_walks, L+1)

    def aggregate_walks(self, walks, node_features):
        """
        walks: (B, num_walks, L+1)
        node_features: (N, in_dim)
        returns: aggregated features per source node: (B, in_dim)
        """
        B = walks.size(0)
        k = self.num_walks
        Lp1 = walks.size(2)
        # gather features for each index
        # reshape to (B * k * (L+1),)
        flat_idxs = walks.reshape(-1)  # (B*k*(L+1),)
        gathered = node_features[flat_idxs]  # (B*k*(L+1), in_dim)
        gathered = gathered.reshape(B, k, Lp1, -1)  # (B, k, L+1, in_dim)

        if self.aggregator == 'mean':
            # mean across steps then mean across walks
            step_mean = gathered.mean(dim=2)     # (B, k, in_dim)
            walk_mean = step_mean.mean(dim=1)    # (B, in_dim)
            return walk_mean
        elif self.aggregator == 'sum':
            return gathered.sum(dim=(1,2))  # (B, in_dim)
        elif self.aggregator == 'rnn':
            # flatten walks along batch: (B*k, L+1, in_dim)
            bc = gathered.reshape(B * k, Lp1, -1)
            out, h = self.gru(bc)  # h: (1, B*k, in_dim)
            h = h.squeeze(0).reshape(B, k, -1)  # (B, k, in_dim)
            return h.mean(dim=1)
        else:
            raise ValueError("Unknown aggregator")

    def forward(self, source_nodes, node_features):
        """
        source_nodes: LongTensor (B,) indices for which to compute outputs
        node_features: (N, in_dim)
        returns: (B, out_dim)
        """
        # sample walks for source nodes
        walks = self.sample_walks(source_nodes)  # (B,k,L+1)
        agg = self.aggregate_walks(walks, node_features)  # (B, in_dim)
        out = self.mlp(agg)  # (B, out_dim)
        return out

class RandomWalkGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, P, num_walks=8, walk_length=4, device='cpu'):
        super().__init__()
        self.layer1 = RandomWalkLayer(in_dim, hidden_dim, P,
                                      num_walks=num_walks, walk_length=walk_length, aggregator='mean', device=device)
        # second layer uses node features from previous layer for all nodes;
        # to avoid sampling from whole set for intermediate representations, we can compute
        # representations only for batch nodes or precompute for all nodes each epoch.
        self.layer2 = RandomWalkLayer(hidden_dim, out_dim, P,
                                      num_walks=num_walks, walk_length=walk_length, aggregator='mean', device=device)
        self.device = device

    def forward(self, batch_nodes, all_node_features):
        """
        batch_nodes: indices of nodes in the minibatch for which we compute outputs (B,)
        all_node_features: (N, in_dim) - raw node features
        """
        # Option A: compute h1 for all nodes once (costly) and then sample walks using h1 features.
        # Option B: compute h1 only for batch nodes and use sampling on raw features (less consistent).
        # Here we'll compute h1 for all nodes (feasible for small-medium graphs) for correctness.

        N = all_node_features.size(0)
        # compute h1 for all nodes by feeding each node as source in layer1
        all_indices = torch.arange(N, device=self.device)
        h1 = self.layer1(all_indices, all_node_features)  # (N, hidden_dim)

        # now for batch nodes compute final outputs using h1 as node features
        out = self.layer2(batch_nodes, h1)  # (B, out_dim)
        return out

# Example training loop (node classification)
def train_example(adj, X, labels, train_idx, val_idx, num_epochs=50, batch_size=64, device='cpu'):
    N, F = X.shape
    num_classes = int(labels.max().item() + 1)
    P = row_normalize_adj(adj).to(device)

    model = RandomWalkGNN(in_dim=F, hidden_dim=64, out_dim=num_classes, P=P,
                          num_walks=8, walk_length=4, device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    X_device = X.to(device)
    labels_device = labels.to(device)

    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(train_idx.size(0))
        for i in range(0, perm.size(0), batch_size):
            batch_nodes = train_idx[perm[i:i+batch_size]]
            logits = model(batch_nodes, X_device)  # (B, C)
            loss = crit(logits, labels_device[batch_nodes])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Optional validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_idx, X_device)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == labels_device[val_idx]).float().mean().item()
        if epoch % 5 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch:03d} loss={loss.item():.4f} val_acc={val_acc:.4f}")

    return model
