import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_uniform_
from models.base import KGModel
from models.messagepassing import MessagePassing, scatter_, BaseGNN
from models.gnnbase import GNN
from utils.euclidean import euc_sqdistance, givens_reflection, multi_bmm, givens_unitary, multi_index_select, norm_clamp
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, hyp_distance, logmap0, tanh
from utils.hyperbolic import lorentz_boost, expmap0_lorentz, hyp_distance_multi_c_lorentz, logmap0_lorentz, explicit_lorentz
from .mlp import MLP

EUC_GNN_MODELS = ["CompGCN"]

class CompGCNConv(MessagePassing):
    def __init__(self, opn="add", **kwargs):
        super(CompGCNConv, self).__init__(**kwargs)
        assert opn in {'add', 'mult'}, 'Composition function must be add or mult'
        self.opn = opn

        self.w_loop = nn.Parameter(torch.randn(self.in_channels, self.out_channels, dtype=self.data_type))
        self.w_in = nn.Parameter(torch.randn(self.in_channels, self.out_channels, dtype=self.data_type))
        self.w_out = nn.Parameter(torch.randn(self.in_channels, self.out_channels, dtype=self.data_type))
        self.w_rel = nn.Linear(self.in_channels, self.out_channels, bias=False, dtype=self.data_type)
        self.loop_rel = nn.Parameter(torch.randn(1, self.in_channels, dtype=self.data_type))
        self.bn	= nn.BatchNorm1d(self.out_channels)

        with torch.no_grad():
            xavier_uniform_(self.w_loop)
            xavier_uniform_(self.w_in)
            xavier_uniform_(self.w_out)
            xavier_uniform_(self.w_rel.weight)

    def forward(self, x, edge_index, edge_type, rel_embed):
        out = self.propagate(edge_index, x=x, edge_type=edge_type, rel_embed=rel_embed)
        out = self.bn(out)
        if not self.act is None:
            out = self.act(out)
        out_rel = self.w_rel(rel_embed)
        return out, out_rel
    
    def propagate(self, edge_index, x, edge_type, rel_embed):
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type,  out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]
        loop_index  = torch.arange(num_ent).to(self.device)
        out_inward = self.message(
            x[in_index[1]], in_type, rel_embed, "in"
        )
        out_outward = self.message(
            x[out_index[1]], out_type, rel_embed, "out"
        )
        out_loop = self.message(
            x[loop_index], None, None, "loop"
        )

        # Aggregate
        edge_norm = self.compute_norm(in_index, x.size(0), drop=False).unsqueeze(1)
        out_inward = edge_norm * out_inward
        out_inward = scatter_("add",
                    out_inward,
                    in_index[0],
                    dim_size=x.size(0))
        edge_norm = self.compute_norm(out_index, x.size(0), drop=False).unsqueeze(1)
        out_outward = edge_norm * out_outward
        out_outward = scatter_("add",
                    out_outward,
                    out_index[0],
                    dim_size=x.size(0))

        # Update
        out = (1/3) * self.drop(out_inward) + (1/3) * self.drop(out_outward) + (1/3) * out_loop
        del in_index, in_type, out_index, out_type, loop_index
        return out

    def rel_transform(self, x, rel_embed):
        if self.opn == "add":
            trans_embed = x - rel_embed
        elif self.opn == "mult":
            trans_embed = x * rel_embed
        return trans_embed
    
    def message(self, x_j, edge_type, rel_embed, mode):
        if mode == "in":
            w = self.w_in
        elif mode == "out":
            w = self.w_out
        elif mode == "loop":
            w = self.w_loop
            edge_type = self.loop_rel
        else:
            raise ValueError("Message mode must be in, out or loop")
        if mode != "loop":
            r = torch.index_select(rel_embed, 0, edge_type)
        else:
            r = self.loop_rel

        x = self.rel_transform(x_j, r)
        return x @ w

class CompGCNBase(BaseGNN):
    def __init__(self, opn="add", **kwargs):
        super(CompGCNBase, self).__init__(**kwargs)
        for l in self.layers:
            l.opn = opn
        self.drop_in_between=True

class CompGCN(GNN):
    def __init__(self, args, dataset):
        super(CompGCN, self).__init__(args, dataset)
        self.B = args.basis if args.basis else 0

        if self.B > 0:
            del self.rel
            # Basis vectors
            self.rel_diag = nn.Embedding(self.B, self.rank)
            # Coefficients
            self.rel = nn.Embedding(self.sizes[1], self.B)

        self.base = CompGCNBase(
            opn=args.opn if args.opn else "mult",
            in_channels=self.rank,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            in_channels_r=self.rank,
            hidden_channels_r=self.hidden_dim,
            out_channels_r=self.hidden_dim,
            layers=args.layers,
            act=tanh,
            act_r=nn.Identity(),
            mp=CompGCNConv,
            dropout=args.dropout,
            dtype=args.dtype
        )
        self.interaction = args.interaction.lower() if args.interaction else "distmult"
        assert self.interaction in {'distmult', 'transe'}, 'Invalid interaction model'
    
    def get_r(self):
        if self.B > 0:
            return self.rel.weight @ self.rel_diag.weight
        else:
            return self.rel.weight
    
    def get_queries(self, queries, cache=None):
        if cache is None:
            x, r = self.forward_base()
        else:
            x, r = cache
        # Decoder here
        head = multi_index_select(x, queries[..., 0])
        rel = multi_index_select(r, queries[..., 1])
        if self.interaction == 'distmult':
            lhs_e = head * rel
        elif self.interaction == 'transe':
            lhs_e = head + rel

        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases
    
    def similarity_score(self, lhs_e, rhs_e):
        # print(lhs_e.shape, rhs_e.shape)
        if self.interaction == 'distmult':
            scores = (lhs_e * rhs_e).sum(dim=-1, keepdim=True)
        elif self.interaction == 'transe':
            scores = - euc_sqdistance(lhs_e, rhs_e)
        # print(scores.shape)
        return scores
    




        


