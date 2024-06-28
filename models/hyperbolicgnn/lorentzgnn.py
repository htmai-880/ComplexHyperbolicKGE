import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter, scatter_add
from torch.nn.init import xavier_normal_, xavier_uniform_, kaiming_uniform_
from models.base import KGModel
from models.messagepassing import MessagePassing, scatter_, BaseGNN
from models.gnnbase import GNN
from utils.euclidean import givens_rotations, givens_reflection, multi_bmm, givens_unitary, multi_index_select, norm_clamp
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, hyp_distance, logmap0, tanh
from utils.hyperbolic import lorentz_boost, expmap0_lorentz, hyp_distance_multi_c_lorentz, logmap0_lorentz, explicit_lorentz
from ..mlp import MLP
from .hyperbolicgnn import HyperbolicBase

LORENTZ_GNN_MODELS = ["LorentzGCN"]

class LorentzConv(MessagePassing):
    def __init__(self, **kwargs):
        super(LorentzConv, self).__init__(**kwargs)
        self.w_loop = nn.Parameter(torch.randn(1, self.in_channels, self.out_channels, dtype=self.data_type))
        self.w_in = nn.Parameter(torch.randn(1, self.in_channels, self.out_channels, dtype=self.data_type))
        self.w_out = nn.Parameter(torch.randn(1, self.in_channels, self.out_channels, dtype=self.data_type))

        self.b_loop = nn.Parameter(torch.randn(1, self.out_channels, dtype=self.data_type))
        self.b_in = nn.Parameter(torch.randn(1, self.out_channels, dtype=self.data_type))
        self.b_out = nn.Parameter(torch.randn(1, self.out_channels, dtype=self.data_type))

        self.w_rel = nn.Linear(3*self.in_channels+1, 3*self.out_channels, bias=True, dtype=self.data_type)
        self.w_activation = nn.Linear(self.in_channels, self.in_channels, bias=False, dtype=self.data_type)
        # self.w_scale = nn.Linear(3*self.in_channels, self.out_channels, bias=True, dtype=self.data_type)
        self.loop_curvature = nn.Parameter(torch.Tensor(1).to(self.data_type)) # Used as the inner curvature unless specified otherwise

        self.b_rel1 = nn.Parameter(torch.randn(1, self.out_channels, dtype=self.data_type))
        self.b_rel2 = nn.Parameter(torch.randn(1, self.out_channels, dtype=self.data_type))

        self.loop_weight = nn.Parameter(torch.zeros(1, dtype=self.data_type))

        self.mlp_curvature = MLP(3*self.in_channels+1,  3*self.in_channels, 1, num_layers=2, dtype=self.data_type)

        with torch.no_grad():
            nn.init.ones_(self.loop_curvature)
            self.w_activation.weight = nn.Parameter(torch.eye(self.in_channels, dtype=self.data_type))
            xavier_uniform_(self.w_loop)
            xavier_uniform_(self.w_in)
            xavier_uniform_(self.w_out)
            nn.init.zeros_(self.b_loop)
            nn.init.zeros_(self.b_in)
            nn.init.zeros_(self.b_out)
            nn.init.zeros_(self.b_rel1)
            nn.init.zeros_(self.b_rel2)
            
    def forward(self, x, edge_index, edge_type, rel_embed):
        if self.device is None:
            self.device = x.device

        rel_embed, curvatures = rel_embed
        trans_rot_c_r = rel_embed[..., :3 * self.in_channels]
        trans_rot_c_r = torch.cat([trans_rot_c_r, curvatures], dim=-1)
        out_rel = self.w_rel(trans_rot_c_r)
        # The new curvature should be applicable to the newly formed relation embeddings.
        curvatures_out = self.mlp_curvature(trans_rot_c_r)
        curvatures_out_ = F.softplus(curvatures_out)

        # out_scale = tanh(self.w_scale(trans_rot_r))
        # out_scale = out_scale + torch.ones_like(out_scale)
        # out_rel = torch.cat([out_rel, out_scale], dim=-1)

        # rel_embed is of dimension (N, d)
        out = self.propagate(edge_index,   x=x, edge_type=edge_type,   rel_embed=out_rel, curvatures=curvatures_out_)
        # Nonlinearity
        if not self.act is None:
            out = self.act(out)

        # Dropout
        out = self.drop(out)
        out_rel = self.drop(out_rel)
        
        return out, out_rel, curvatures_out

    def propagate(self, edge_index, x, edge_type, rel_embed, curvatures=None):
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        loop_curvature = F.softplus(self.loop_curvature)
        if curvatures is None:
            # Use loop curvature as unique curvature
            curvatures = loop_curvature
        
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type,  out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

        loop_index  = torch.arange(num_ent).to(self.device)
        # loop_type   = torch.full((num_ent,), loop_relation, dtype=torch.long).to(self.device)

        # Lookup for tail entities in edges
        out_inward = self.message(
            x[in_index[1]], out_type, rel_embed, curvatures, "in"
        ).squeeze(1)
        out_outward = self.message(
            x[out_index[1]], in_type, rel_embed, curvatures, "out"
        ).squeeze(1)
        out_loop = self.message(
            x[loop_index], None, None, None, "loop"
        ).squeeze(1)
        # out = torch.cat([out_inward, out_outward, out_loop], dim=0) # [E+N, D]
 
        # edge_index_ = torch.cat([edge_index, loop_index.repeat(2, 1)], dim=1) # [2, E+N]

        # Methods : [1, 2, 3]
        method = 1


        # Representation of tail entities obtained. size: [E + N, D]
        # edge_norm is of dimension [E + N, 1]
        
        # METHOD 1: Aggregation in the tangent space.
        if method == 1:
            out = torch.cat([out_inward, out_outward], dim=0)
            edge_norm = self.compute_norm(edge_index, x.size(0), drop=False).unsqueeze(1)
            loop_weight = F.sigmoid(self.loop_weight)
            out = edge_norm * out # (E, D) for neighbors
            # deal with self-loops now
            out = scatter_("add",
                        out,
                        edge_index[0],
                        dim_size=x.size(0)) # (N, D)

            out = explicit_lorentz(expmap0_lorentz(out, loop_curvature), loop_curvature)
            assert not torch.isnan(out).any(), "xj_rel contains nan values."

            out_loop = explicit_lorentz(expmap0_lorentz(out_loop, loop_curvature), loop_curvature)
            assert not torch.isnan(out_loop).any(), "xj_rel contains nan values."
            # Lorentz centroid
            out = (1 - loop_weight) * out + loop_weight * out_loop
            out_L = - out[..., :1]**2 + torch.sum(out[..., 1:]**2, dim=-1, keepdim=True)
            out_L =(1/(loop_curvature ** 0.5)) * torch.sqrt(torch.abs(out_L)) + 1e-6
            out = (out / out_L)[..., 1:]
            assert not torch.isnan(out).any(), "xj_rel contains nan values."


            out = logmap0_lorentz(out, loop_curvature)
            assert not torch.isnan(out).any(), "xj_rel contains nan values."


        # Output of size [N, D] and lives in the tangent plane
        # METHOD 2: Aggregation in the hyperbolic space using the Lorentz midpoint.
        elif method == 2:
            out = torch.cat([out_inward, out_outward, out_loop], dim=0) # [E+N, D]
            edge_index_ = torch.cat([edge_index, loop_index.repeat(2, 1)], dim=1) # [2, E+N]
            edge_norm = self.compute_norm(edge_index_, x.size(0), drop=False).unsqueeze(1)
            out = self.update(out, edge_norm, edge_index_, loop_curvature, num_ent)

        elif method == 3:
            out = torch.cat([out_inward, out_outward, out_loop], dim=0) # [E+N, D]
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
            
            out = (1/3) * out_inward + (1/3) * out_outward + (1/3) * out_loop

        del in_index, in_type, out_index, out_type, loop_index
        return out

    def get_regularizable_params(self):
        return [
            self.w_loop,
            self.w_in,
            self.w_out,
            self.w_rel.weight
        ]

    def rel_transform(self, ent_embed, rel_embed, curvatures):
        # Parametrize isometry in the same fashion as in RotH.
        # The relation embedding should be 3 times the entity embedding.
        # rel1, rel2, rot, scale = torch.chunk(rel_embed, 4, dim=-1)
        rel1, rel2, rot = torch.chunk(rel_embed, 3, dim=-1)
        # scale1, scale2 = torch.chunk(scale, 2, dim=-1)
        lhs = expmap0_lorentz(ent_embed, curvatures)
        assert not torch.isnan(lhs).any(), "xj_rel contains nan values."
        # Add
        lhs = lorentz_boost(lhs, rel1, curvatures)
        assert not torch.isnan(lhs).any(), "xj_rel contains nan values."
        # Rotate (note: this is an isometry, so no need to project to the tangent plane)
        # lhs = logmap0_lorentz(lhs, curvatures)
        # lhs = givens_rotations(rot, lhs)
        # lhs[..., 0::2].mul_(1/scale2)
        # lhs[..., 1::2].mul_(1/scale2)
        # lhs = givens_rotations(rot, lhs, scale=scale1, inverse=True)
        lhs = givens_rotations(rot, lhs, scale=None, inverse=False)
        assert not torch.isnan(lhs).any(), "xj_rel contains nan values."

        # lhs = expmap0_lorentz(lhs, curvatures)
        # Add again
        lhs = lorentz_boost(lhs, rel2, curvatures)
        assert not torch.isnan(lhs).any(), "xj_rel contains nan values."
        return logmap0_lorentz(lhs, curvatures)
    
    def message(self, x_j, edge_type, rel_embed, curvatures, mode):
        # x_j is the neighbor embeddings
        # edge_type is the type of the edge
        # rel_embed is the embeddings of the relations
        # edge_norm is the normalization factor for the edges
        # mode is the direction of the edges
        # Will use the inverse from the inverse relationship directly.
        weight = getattr(self, 'w_{}'.format(mode))
        x_j = x_j.unsqueeze(-2).unsqueeze(-2) # (E, 1, 1, D)
        assert not torch.isnan(x_j).any(), "xj contains nan values."
        x_j = (x_j @ weight).squeeze(-2).squeeze(-2) # (E, d)
        assert not torch.isnan(x_j).any(), "xj contains nan values."
        loop_curvature = F.softplus(self.loop_curvature)
        x_j = expmap0_lorentz(x_j, loop_curvature)
        bias = getattr(self, 'b_{}'.format(mode))
        assert not torch.isnan(x_j).any(), "xj_rel contains nan values."
        x_j = lorentz_boost(x_j, bias, loop_curvature)
        assert not torch.isnan(x_j).any(), "xj_rel contains nan values."
        x_j = logmap0_lorentz(x_j, loop_curvature)
        assert not torch.isnan(x_j).any(), "xj_rel contains nan values."
        if mode != "loop":
            rel_c = torch.index_select(curvatures, 0, edge_type) if curvatures.nelement() > 1 else curvatures
            rel_emb = torch.index_select(rel_embed, 0, edge_type)
            x_j  = self.rel_transform(x_j, rel_emb, rel_c) # (E, D)
        assert not torch.isnan(x_j).any(), "xj_rel contains nan values."
        return x_j


class LorentzGCN(GNN):
    def __init__(self, args, dataset):
        super(LorentzGCN, self).__init__(args, dataset)
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        # self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.multi_c = args.multi_c
        self.c_layer = nn.Embedding(self.sizes[1], 1)

        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

        self.base = HyperbolicBase(
            in_channels=self.rank,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            in_channels_r=3*self.rank,
            hidden_channels_r=3*self.hidden_dim,
            out_channels_r=3*self.hidden_dim,
            layers=args.layers,
            act=tanh,
            act_r=tanh,
            mp=LorentzConv,
            dropout=args.dropout,
            dtype=args.dtype
        )

    def get_r(self):
        r = torch.cat((self.rel.weight, self.rel_diag.weight), dim=-1)
        c = self.c_layer.weight # (N_r, 1)
        return (r, c)

    
    def forward_base(self):
        # Use the forward_base from the super class
        x, (r, c) = super().forward_base()

        c = F.softplus(c)
        if not self.multi_c:
            c = c.mean(dim=0, keepdim=True)
        return x, (r, c)
    
    def get_queries(self, queries, cache=None):
        if cache is None:
            x, (r, curvatures) = self.forward_base()
        else:
            x, (r, curvatures) = cache
        r = multi_index_select(r, queries[..., 1])
        rel1, rel2, rot = torch.chunk(r, 3, dim=-1)
        # rel1, rel2, rot, scale = torch.chunk(r, 4, dim=-1)
        # scale1, scale2 = torch.chunk(scale, 2, dim=-1)

        """Compute embedding and biases of queries."""
        c = multi_index_select(curvatures[..., -1:], queries[..., 1]) if self.multi_c else curvatures
        head = expmap0_lorentz(
            multi_index_select(x, queries[..., 0]),
            c
        )   # hyperbolic

        lhs = lorentz_boost(head, rel1, c)   # hyperbolic
        # lhs = logmap0(lhs, c)
        res1 = givens_rotations(rot, lhs, scale=None)
        # res1 = givens_rotations(rot, lhs, scale=scale1)   # givens_rotation(Euclidean, hyperbolic)
        # res1[..., 0::2].mul_(scale2)
        # res1[..., 1::2].mul_(scale2)
        # res1 = expmap0(res1, c)
        res2 = lorentz_boost(res1, rel2, c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        rhs_e = expmap0_lorentz(rhs_e, c)
        dist = hyp_distance_multi_c_lorentz(lhs_e, rhs_e, c)
        return - dist ** 2