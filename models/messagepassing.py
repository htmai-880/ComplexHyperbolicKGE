
import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch_scatter import scatter, scatter_add


def get_param(shape, gain=1.0):
    param = nn.Parameter(torch.Tensor(*shape))
    if len(shape) >= 2:
        xavier_normal_(param.data, gain=gain)
    return param

def scatter_(name, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out

class MessagePassing(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_r, out_channels_r, act=None, dropout=0.0, dtype=None,
                 **kwargs):
        super(MessagePassing, self).__init__()
        if dtype is None:
            self.data_type = torch.float
        elif dtype == 'double':
            self.data_type = torch.double
            # self.bias_type = torch.double
        elif dtype == 'float':
            self.data_type = torch.float

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_r = in_channels_r
        self.out_channels_r = out_channels_r
        self.act = act
        self.drop = torch.nn.Dropout(dropout)
        self.device = None
    
    def forward(self, x, edge_index, edge_type, rel_embed):
        return x, rel_embed
    
    def propagate(self, aggr, edge_index, **kwargs):
        pass
    
    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""
        return x_j
    
    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""
        return aggr_out
    
    def compute_norm(self, edge_index, num_ent, drop=False):
        row, _	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        if drop:
            edge_weight = self.drop(edge_weight)
        deg		= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv		= deg.pow(-1)
        deg_inv[deg_inv	== float('inf')] = 0
        norm		= deg_inv[row] * edge_weight
        return norm

    def compute_symmetric_norm(self, edge_index, num_ent, drop=False):
        row, col	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        if drop:
            edge_weight = self.drop(edge_weight)
        deg		= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv		= deg.pow(-0.5)
        deg_inv[deg_inv	== float('inf')] = 0
        norm		= deg_inv[row] * edge_weight * deg_inv[col]
        return norm
    
    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels)
    

class BaseGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 in_channels_r, hidden_channels_r, out_channels_r,
                 layers : int,
                 act, act_r,
                 mp,
                 dropout=0.0,
                 dtype=None,
                 kwargs_first_layer=None,
                 kwargs_hidden_layer=None,
                 kwargs_last_layer=None
                 ):
        super(BaseGNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels_r = in_channels_r
        self.out_channels_r = out_channels_r
        self.hidden_channels = hidden_channels
        self.hidden_channels_r = hidden_channels_r
        self.act = act
        self.act_r = act_r
        self.layers = nn.ModuleList()
        self.mp_class = mp
        self.n_layers = layers
        self.dropout = dropout
        self.dtype = dtype
        self.kwargs_first_layer = kwargs_first_layer if isinstance(kwargs_first_layer, dict) else {}
        self.kwargs_hidden_layer = kwargs_hidden_layer  if isinstance(kwargs_hidden_layer, dict) else {}
        self.kwargs_last_layer = kwargs_last_layer  if isinstance(kwargs_last_layer, dict) else {}

        if self.n_layers == 1:
            self.layers.append(self.make_first_layer())
        else:
            self.layers.append(self.make_first_layer())
            for _ in range(self.n_layers - 2):
                self.layers.append(self.make_hidden_layer())
            self.layers.append(self.make_last_layer())
    
    def make_first_layer(self):
        if self.n_layers == 1:
            return self.mp_class(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                in_channels_r = self.in_channels_r,
                out_channels_r = self.out_channels_r,
                act = None,
                dropout = 0.0,
                dtype = self.dtype,
                **self.kwargs_first_layer
            )
        else:
            return self.mp_class(
                in_channels = self.in_channels,
                out_channels = self.hidden_channels,
                in_channels_r = self.in_channels_r,
                out_channels_r = self.hidden_channels_r,
                act = self.act,
                dropout = self.dropout,
                dtype = self.dtype,
                **self.kwargs_first_layer
            )
    
    def make_hidden_layer(self):
        return self.mp_class(
            in_channels = self.hidden_channels,
            out_channels = self.hidden_channels,
            in_channels_r = self.hidden_channels_r,
            out_channels_r = self.hidden_channels_r,
            act = self.act,
            dropout = self.dropout,
            dtype = self.dtype,
            **self.kwargs_hidden_layer
        )
    
    def make_last_layer(self):
        return self.mp_class(
            in_channels = self.hidden_channels,
            out_channels = self.out_channels,
            in_channels_r = self.hidden_channels_r,
            out_channels_r = self.out_channels_r,
            act = None,
            dropout = 0.0,
            dtype = self.dtype,
            **self.kwargs_last_layer
        )

    def forward(self, x, edge_index, edge_type, rel_embed):
        for i, layer in enumerate(self.layers):
            x, rel_embed = layer(x, edge_index, edge_type, rel_embed)
            if i != len(self.layers) - 1:
                rel_embed = self.act_r(rel_embed)
        return x, rel_embed
    