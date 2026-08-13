import torch

def anonymous_walk(walk):
    anon_walk = torch.zeros(walk.shape, dtype=walk.dtype)
    for i, walk_i in enumerate(walk):
        mapping = {}
        anon_list = torch.zeros(walk.shape[1], dtype=walk.dtype)
        id = 0
        for j, node in enumerate(walk_i):
            if node.item() not in mapping:
                mapping[node.item()] = id
                id += 1
            anon_list[j] = mapping[node.item()]
        anon_walk[i] = anon_list

    return anon_walk
