import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import math

class MonotonicLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.abs(self.weight), self.bias)


    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
class MonotonicMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, dtype):
        super(MonotonicMLP, self).__init__()
        self.act = F.relu
        self.layers = nn.ModuleList()
        self.layers.append(MonotonicLinear(in_features, hidden_dim, bias=True, dtype=dtype))
        self.layers.append(MonotonicLinear(hidden_dim, out_features, bias=True, dtype=dtype))
    
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers)-1:
                out = self.act(out)
        return out
