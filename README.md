REQUIREMENTS:
    - pytorch
    - pytorch geometric
    - ogb
    - matplotlib

-----------------------------------------------------------------------------------------------------------------------------
anonymous walks: https://people.csail.mit.edu/silvio/Selected%20Scientific%20Papers/Reconstructing%20Markov%20Processes%20from%20Independent%20and%20Ananymous%20Experiments%20(Published%20Version).pdf
    - original i think

anonymous walk embeddings: https://proceedings.mlr.press/v80/ivanov18a/ivanov18a.pdf
    - to find a feature representation for the AW itself

beyond message passing: https://arxiv.org/pdf/2501.18739
    - sequence embeddings seem nice (both anonymous and random), transformer part idc

message passing all the way up: https://arxiv.org/pdf/2202.11097
    - may be useful for proofs etc didn't read

node-edge co-embedding: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9224195
    - alternating between node and edge layers (line graph)
    - can it be applied to "node complement" subgraph?

message passing with loops: file:///home/noyan/Downloads/cantwell-newman-2019-message-passing-on-networks-with-loops.pdf#page=2.52

WL graph kernels: https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
    - graph classification kernel i believe

Graph Elimination Network: https://arxiv.org/pdf/2401.01233
    - an MPNN variant that approximates graph transformer's long range understanding
    - self attention on a k-hop receptive field for each vertex, and an edge-wise self attention

GAT: https://arxiv.org/pdf/1710.10903
    - balls

removal based node influence: https://dl.acm.org/doi/pdf/10.1145/3589334.3645389

tutorial on WL: https://arxiv.org/pdf/2201.07083
    - for proofs if needed (characterization of the expressive power of the model/approach)

-----------------------------------------------------------------------------------------------------------------------------
IDEAS:
    - calculate k-hop attention features on the node complement subgraph for each neighbor, and aggregate them for the embedding of the node in question (TODO: implement GEA elimination message propagation (eqs. 1->11))
    - as a second loss, consider the difference between embeddings of neighbor vertices (e.g. euclidean distance) to be minimized. we want to enforce that neighbor nodes are similar. but at the same time, we'd need some additional mechanism to ensure that different "clusters" or class accumulations will result in different class output, even though they are neighbors idk
    - GEN paper eqn. 16, normalization after every layer, may be helpful to avoid over-smoothing (?). the transformer idea also seems nice, allowing the model to decide on the importance of the aggregated embedding result
    - subgraph sampling?
    - isn't the problem at hand literally a simple markov process (not even MDP)? meaning we can solve it using the bellman eqn. we don't even need an iterative approach since the transition probabilities are known, we should be able to solve it idk am i delulu
