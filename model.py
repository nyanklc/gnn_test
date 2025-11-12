import torch
import torch.nn as nn
import torch.functional as F

class BellmanRNN(nn.RNN):
    def __init__(self, feature_size, num_layers):
        super().__init__(feature_size, feature_size, num_layers)

    def forward(self, x):
        super().forward(x)
