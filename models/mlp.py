import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dtype=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
        self.layers.append(nn.Linear(hidden_dim, output_dim, dtype=dtype))
        self.num_layers = num_layers
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers-1:
                x = F.relu(x)
        return x
