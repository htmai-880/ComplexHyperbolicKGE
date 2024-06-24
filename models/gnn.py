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
from .mlp import MLP



GNN_MODELS = ["PoincareGCN", "LorentzGCN", "PoincareGAT"]


class PoincareConv(MessagePassing):
    def __init__(self, **kwargs):
        super(PoincareConv, self).__init__(**kwargs)
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
            kaiming_uniform_(self.w_loop)
            kaiming_uniform_(self.w_in)
            kaiming_uniform_(self.w_out)
            kaiming_uniform_(self.b_loop)
            kaiming_uniform_(self.b_in)
            kaiming_uniform_(self.b_out)

            kaiming_uniform_(self.b_rel1)
            kaiming_uniform_(self.b_rel2)
            
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



        # Will perform a Möbius addition for the relation embeddings

        rel1, rel2, rot = torch.chunk(out_rel, 3, dim=-1)
        rel1 = expmap0(rel1, curvatures_out_)
        rel2 = expmap0(rel2, curvatures_out_)
        b_rel1 = expmap0(self.b_rel1, curvatures_out_)
        b_rel2 = expmap0(self.b_rel2, curvatures_out_)
        rel1 = mobius_add(rel1, b_rel1, curvatures_out_)
        rel2 = mobius_add(rel2, b_rel2, curvatures_out)
        rel1 = logmap0(rel1, curvatures_out_)
        rel2 = logmap0(rel2, curvatures_out_)

        out_rel = torch.cat([rel1, rel2, rot], dim=-1)

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

        # out_rel = tanh(out_rel)
        
        return out, (out_rel, curvatures_out)

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
            x[in_index[1]], in_type, rel_embed, curvatures, "in"
        ).squeeze(1)
        out_outward = self.message(
            x[out_index[1]], out_type, rel_embed, curvatures, "out"
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
        # if the model is the Poincaré ball, out is in the tangent space.
        # We can perform aggregation in the tangent space
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

            # # Do gyrobarycenter
            # out = expmap0(out, loop_curvature)
            # out_loop = expmap0(out_loop, loop_curvature)
            # gamma_rel = torch.sum(out * out, dim=-1, keepdim=True)
            # gamma_rel = 2 / (1 - loop_curvature * gamma_rel)
            # rel_weight = 1 - loop_weight

            # gamma_loop = torch.sum(out_loop * out_loop, dim=-1, keepdim=True)
            # gamma_loop = 2 / (1 - loop_curvature * gamma_loop)
            # # loop weight defined above
            # den = rel_weight * (gamma_rel - 1) + loop_weight * (gamma_loop - 1)

            # rel_weight = rel_weight * gamma_rel / den
            # loop_weight = loop_weight * gamma_loop / den
            # out = rel_weight * out  + loop_weight * out_loop
            # factor = torch.sqrt(1 - loop_curvature * torch.sum(out * out, dim=-1, keepdim=True))
            # factor = 1 / (1 + factor)
            # out = factor * out
            # out = logmap0(out, loop_curvature)

            # out = (1 - loop_weight) * out + loop_weight * out_loop

            # Compute degrees
            degs = torch.ones_like(edge_norm)
            degs = scatter_("add", degs, edge_index[0], dim_size=x.size(0)).squeeze(1) # (N,)
            out_ = torch.zeros(out.size(0), out.size(1), dtype=out.dtype).to(self.device)
            out_[degs == 0] = out_loop[degs == 0]

            out = expmap0((1 - loop_weight) * out[degs > 0], loop_curvature)
            out_loop = expmap0(loop_weight * out_loop[degs > 0], loop_curvature)
            out = project(
                mobius_add(out, out_loop, loop_curvature),
                loop_curvature
            )
            out = logmap0(out, loop_curvature)
            out_[degs > 0] = out
            out = out_

        # Output of size [N, D] and lives in the tangent plane
        # METHOD 2: Aggregation in the hyperbolic space using gyromidpoint.
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

    def update(self, out, edge_norm, edge_index_, loop_curvature, num_ent):
        out = expmap0(out, loop_curvature) # (E+N, D)
        # Compute metric
        weights = 2 / (1 - loop_curvature * torch.sum(out * out, dim=-1, keepdim=True)) # (E+N, 1)
        den = edge_norm * (weights-1) # (E + N, 1)
        den = scatter_("add", den, edge_index_[0], dim_size=num_ent) # (N, 1)
        den = den[edge_index_[0]] + 1e-5 # (E+N, 1)
        weights = weights * edge_norm / den # (E+N, 1)
        out = weights * out # (E+N, D)
        out = scatter_("add",
            out,
            edge_index_[0],
            dim_size=num_ent
        ) # (N, D)
        # Einstein scalar mul
        # See https://arxiv.org/pdf/2006.08210 (D.5)
        factor = torch.sqrt(1 - loop_curvature * torch.sum(out * out, dim=-1, keepdim=True))
        factor = 1 / (1 + factor)
        out = factor * out
        out = logmap0(out, loop_curvature)
        return out

    def rel_transform(self, ent_embed, rel_embed, curvatures):
        # Parametrize isometry in the same fashion as in RotH.
        # The relation embedding should be 3 times the entity embedding.
        # rel1, rel2, rot, scale = torch.chunk(rel_embed, 4, dim=-1)
        rel1, rel2, rot = torch.chunk(rel_embed, 3, dim=-1)
        # scale1, scale2 = torch.chunk(scale, 2, dim=-1)
        lhs = expmap0(ent_embed, curvatures)
        rel1 = expmap0(rel1, curvatures)   # hyperbolic
        rel2 = expmap0(rel2, curvatures)   # hyperbolic
        # Add
        lhs = project(mobius_add(-rel2, lhs, curvatures), curvatures)
        # Rotate (note: this is an isometry, so no need to project to the tangent plane)
        # lhs = logmap0(lhs, curvatures)
        # lhs = givens_rotations(rot, lhs)
        # lhs[..., 0::2].mul_(1/scale2)
        # lhs[..., 1::2].mul_(1/scale2)
        # lhs = givens_rotations(rot, lhs, scale=scale1, inverse=True)
        lhs = givens_rotations(rot, lhs, scale=None, inverse=True)
        # lhs = expmap0(lhs, curvatures)
        # Add again
        lhs = mobius_add(-rel1, lhs, curvatures)
        return logmap0(lhs, curvatures)


    def message(self, x_j, edge_type, rel_embed, curvatures, mode):
        # x_j is the neighbor embeddings
        # edge_type is the type of the edge
        # rel_embed is the embeddings of the relations
        # edge_norm is the normalization factor for the edges
        # mode is the direction of the edges
        # Will use the inverse from the inverse relationship directly.
        weight = getattr(self, 'w_{}'.format(mode))
        x_j = x_j.unsqueeze(-2).unsqueeze(-2) # (E, 1, 1, D)
        x_j = (x_j @ weight).squeeze(-2).squeeze(-2) # (E, d)
        loop_curvature = F.softplus(self.loop_curvature)
        x_j = expmap0(x_j, loop_curvature)
        bias = getattr(self, 'b_{}'.format(mode))
        bias = expmap0(bias, loop_curvature)
        x_j = project(
            mobius_add(x_j, bias, loop_curvature), loop_curvature
        )
        x_j = logmap0(x_j, loop_curvature)
        if mode != "loop":
            rel_c = torch.index_select(curvatures, 0, edge_type) if curvatures.nelement() > 1 else curvatures
            rel_emb = torch.index_select(rel_embed, 0, edge_type)
            x_j  = self.rel_transform(x_j, rel_emb, rel_c) # (E, D)

        assert not torch.isnan(x_j).any(), "xj_rel contains nan values."
        return x_j


class PoincareGATConv(PoincareConv):
    def __init__(self, gather="mean", **kwargs):
        super(PoincareGATConv, self).__init__(**kwargs)
        self.loop_rel = nn.Parameter(torch.randn(1, 3 * self.out_channels, dtype=self.data_type))
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)
        self.heads = 4

        self.gather = gather
        out_att = self.out_channels if gather=="mean" else self.out_channels // self.heads

        self.w_loop = nn.Parameter(torch.randn(self.heads, self.in_channels, out_att, dtype=self.data_type))
        self.w_in = nn.Parameter(torch.randn(self.heads, self.in_channels, out_att, dtype=self.data_type))
        self.w_out = nn.Parameter(torch.randn(self.heads, self.in_channels, out_att, dtype=self.data_type))

        self.W_r = nn.Parameter(torch.randn(self.heads, 3 * self.out_channels, out_att, dtype=self.data_type))
        self.a_h = nn.Parameter(torch.randn(1, self.heads, out_att, dtype=self.data_type))
        self.a_r = nn.Parameter(torch.randn(1, self.heads, out_att, dtype=self.data_type))
        self.a_t = nn.Parameter(torch.randn(1, self.heads, out_att, dtype=self.data_type))

        with torch.no_grad():
            nn.init.ones_(self.loop_curvature)
            kaiming_uniform_(self.w_loop)
            kaiming_uniform_(self.w_in)
            kaiming_uniform_(self.w_out)
            kaiming_uniform_(self.W_r)
            xavier_normal_(self.a_h)
            xavier_normal_(self.a_r)
            xavier_normal_(self.a_t)
    
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

        # Lookup for tail entities in edges
        out_inward = self.message(
            x[in_index[1]], out_type, rel_embed, curvatures, "in"
        ) # (E, k, d)
        out_outward = self.message(
            x[out_index[1]], in_type, rel_embed, curvatures, "out"
        ) # (E, k, d)
        out_loop = self.message(
            x[loop_index], None, rel_embed, loop_curvature, "loop"
        ) # (N, k, d)
        method = 1
        # METHOD 1: Aggregation in the tangent space.
        # if the model is the Poincaré ball, out is in the tangent space.
        # We can perform aggregation in the tangent space
        if method == 1:
            out = torch.cat([out_inward, out_outward], dim=0) # (E, k, d)
            edge_norm, out = self.compute_norm(edge_index,
                                x.size(0), drop=False,
                                x_i = out_loop, # (N, k, d)
                                x_j = out, # (E, k, d)
                                edge_type = edge_type,
                                rel_embed = rel_embed.unsqueeze(1),
                                curvatures=loop_curvature
                            ) # (E+N, k, 1) and (E+N, k, d), d = D if mean and d = D/k if concat
            # if self.gather == "mean":
            head_index_ = torch.cat([edge_index[0], loop_index], dim=0)
            # One could verify that the weights add up to 1.
            out = edge_norm * out # (E+N, k, d)
            if self.gather == "mean":
                out = torch.mean(out, dim = 1)
            elif self.gather == "concat":
                out = out.reshape(out.size(0), -1)
            # Update
            out = scatter(out, head_index_, dim=0, out=None, dim_size=num_ent, reduce="add")
            del head_index_
        del in_index, out_index, in_type, out_type, loop_index
        return out

    def compute_norm(self, edge_index, num_ent, drop=False, x_i=None, x_j=None, edge_type=None, rel_embed=None, curvatures=None):
        # Input: (E, 1, d) or # (E, K, d)
        # Output: (E, k, 1)
        if x_i is None or x_j is None:
            return super().compute_norm(edge_index, num_ent, drop=drop)
        # Compute attention weights. Not the only way to do it.
        head_entities = edge_index[0] # (E,)
        h_j_tail = torch.cat([x_j, x_i], dim=0) # (E+N, K, d)

        if (not edge_type is None) and (not rel_embed is None):
            # rel embed: (N_r, 1, d_r)
            r = (rel_embed[..., :3*self.out_channels].unsqueeze(-2) @ self.W_r).squeeze(-2) # (N_r, K, d)
            r_self = (self.loop_rel.view(1, 1, 1, -1) @ self.W_r).squeeze(-2) # (1, K, d)
            # h_ij_tail = h_ij_tail + torch.cat([
            #   r[head_entities], r_self.repeat(num_ent, 1, 1)
            # ], dim=0)
        # Compute a_ij
        head_entities = torch.cat([head_entities, torch.arange(num_ent, device=head_entities.device)])
        a_ij = (self.a_h * x_i).sum(dim=-1, keepdim=True)[head_entities] # (E+N, K, 1)
        a_ij = a_ij + (self.a_t * h_j_tail).sum(dim=-1, keepdim=True) # (E+N, K, 1)
        if (not edge_type is None) and (not rel_embed is None):
            r_ij = (self.a_r * r_self).sum(dim=-1, keepdim=True).repeat(num_ent, 1, 1) # (1, K, d) * (1, K, d) -> sum -> (1, K, 1) -> repeat -> (N, K, 1)
            r_ij = torch.cat([
                (self.a_r * r).sum(dim=-1, keepdim=True)[edge_type],
                r_ij
            ], dim=0) # (E+N, K, 1)
            a_ij = a_ij + r_ij
        a_ij = self.leaky(a_ij) # (E+N, K, 1)
        # Substract max to improve numerical stability
        max_ = scatter_("max", a_ij, head_entities, num_ent) # (E+N, n_head)
        max_ = max_[head_entities] # (E+N, n_head)
        # max_ = 10
        a_ij = torch.exp(a_ij - max_)
        sum_ = scatter_("add", a_ij, head_entities, num_ent)# (E+N, n_head)
        a_ij = a_ij / (sum_[head_entities] + 1e-8)
        del head_entities
        return a_ij, h_j_tail # (E+N, K, 1) and # (E+N, K, D)
    
class PoincareBase(BaseGNN):
    def __init__(self, **kwargs):
        super(PoincareBase, self).__init__(**kwargs)
        self.act_r_base = kwargs.get("act_r")
        self.act_r = (lambda x : (self.act_r_base(x[0]), x[1]))

class PoincareGCN(GNN):
    def __init__(self, args, dataset):
        super(PoincareGCN, self).__init__(args, dataset)
    
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        # self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.multi_c = args.multi_c
        self.c_layer = nn.Embedding(self.sizes[1], 1)

        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

        self.base = PoincareBase(
            in_channels=self.rank,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            in_channels_r=3*self.rank,
            hidden_channels_r=3*self.hidden_dim,
            out_channels_r=3*self.hidden_dim,
            layers=2,
            act=tanh,
            act_r=tanh,
            mp=PoincareConv,
            dropout=args.dropout,
            dtype=args.dtype
        )

    def __setattr__(self, name, value):
        if not "edge_" in name:
            super().__setattr__(name, value)
        else:
            nn.Module.__setattr__(self, name, value)

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
            x, r, curvatures = self.forward_base()
        else:
            x, r, curvatures = cache
        r = multi_index_select(r, queries[..., 1])
        rel1, rel2, rot = torch.chunk(r, 3, dim=-1)
        # rel1, rel2, rot, scale = torch.chunk(r, 4, dim=-1)
        # scale1, scale2 = torch.chunk(scale, 2, dim=-1)

        """Compute embedding and biases of queries."""
        c = multi_index_select(curvatures[..., -1:], queries[..., 1]) if self.multi_c else curvatures
        head = expmap0(
            multi_index_select(x, queries[..., 0]),
            c
        )   # hyperbolic
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        lhs = project(mobius_add(rel1, head, c), c)   # hyperbolic
        # lhs = logmap0(lhs, c)
        res1 = givens_rotations(rot, lhs, scale=None)
        # res1 = givens_rotations(rot, lhs, scale=scale1)   # givens_rotation(Euclidean, hyperbolic)
        # res1[..., 0::2].mul_(scale2)
        # res1[..., 1::2].mul_(scale2)
        # res1 = expmap0(res1, c)
        res2 = mobius_add(rel2, res1, c)   # hyperbolic
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
        rhs_e = expmap0(rhs_e, c)
        dist = hyp_distance_multi_c(lhs_e, rhs_e, c) if self.multi_c else hyp_distance(lhs_e, rhs_e, c)
        return - dist ** 2

    
class PoincareGAT(PoincareGCN):
    def __init__(self, args, dataset):
        super(PoincareGAT, self).__init__(args, dataset)
        hidden_dim = args.hidden_dim if args.hidden_dim else self.rank
        self.hidden_dim = hidden_dim
        channels_r = 4 * args.rank
        hidden_channels_r = 4 * hidden_dim
        del self.layers
        self.layers = nn.ModuleList([PoincareGATConv(
            gather="concat",
            in_channels=self.rank, out_channels=hidden_dim,
            in_channels_r = channels_r, out_channels_r = hidden_channels_r,
            act = tanh,
            dropout = args.dropout,
            dtype=args.dtype
        )])
        for _ in range(args.layers-2):
            self.layers.append(PoincareGATConv(
                gather="concat",
                in_channels=hidden_dim, out_channels=hidden_dim,
                in_channels_r = hidden_channels_r, out_channels_r = hidden_channels_r,
                act = tanh,
                dropout = args.dropout,
                dtype=args.dtype
            ))
        self.layers.append(PoincareGATConv(
            gather="mean",
            in_channels= hidden_dim, out_channels = hidden_dim,
            in_channels_r = hidden_channels_r, out_channels_r = hidden_channels_r,
            act = None,
            dropout = args.dropout,
            dtype=args.dtype
        ))



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
            kaiming_uniform_(self.w_loop)
            kaiming_uniform_(self.w_in)
            kaiming_uniform_(self.w_out)
            kaiming_uniform_(self.b_loop)
            kaiming_uniform_(self.b_in)
            kaiming_uniform_(self.b_out)
            kaiming_uniform_(self.b_rel1)
            kaiming_uniform_(self.b_rel2)
            
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
        # if the model is the Poincaré ball, out is in the tangent space.
        # We can perform aggregation in the tangent space
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
        # METHOD 2: Aggregation in the hyperbolic space using gyromidpoint.
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


class LorentzGCN(KGModel):
    def __init__(self, args, dataset):
        super(LorentzGCN, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
    
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.multi_c = args.multi_c
        self.c_layer = nn.Embedding(self.sizes[1], 1)

        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

        # Fetch the triples from the dataset
        train_examples = dataset.get_examples("train")
        self.edge_index = train_examples[:, [0, 2]].t().contiguous()
        self.edge_type = train_examples[:, 1].contiguous()
        # Two layers
        hidden_dim = args.hidden_dim if args.hidden_dim else self.rank
        self.hidden_dim = hidden_dim
        channels_r = 4 * args.rank
        hidden_channels_r = 4 * hidden_dim
        self.layers = nn.ModuleList([LorentzConv(
            in_channels=self.rank, out_channels=hidden_dim,
            in_channels_r = channels_r, out_channels_r = hidden_channels_r,
            act = tanh,
            dropout = args.dropout,
            dtype=args.dtype
        )])
        for _ in range(args.layers-2):
            self.layers.append(LorentzConv(
                in_channels=hidden_dim, out_channels=hidden_dim,
                in_channels_r = hidden_channels_r, out_channels_r = hidden_channels_r,
                act = tanh,
                dropout = args.dropout,
                dtype=args.dtype
            ))

        self.layers.append(LorentzConv(
            in_channels= hidden_dim, out_channels = hidden_dim,
            in_channels_r = hidden_channels_r, out_channels_r = hidden_channels_r,
            act = None,
            dropout = 0,
            dtype=args.dtype
        ))
        self.edge_dropout = nn.Dropout(args.edge_dropout)
        self.dropout_p = args.dropout

    def __setattr__(self, name, value):
        if not "edge_" in name:
            super().__setattr__(name, value)
        else:
            nn.Module.__setattr__(self, name, value)
    
    def forward_base(self):
        # x = tanh(self.entity.weight) # Typically, embeddings from language models will be reduced with an activation.
        x = self.entity.weight
        assert not torch.isnan(x).any(), "x contains nan values."
        r = torch.cat((self.rel.weight, self.rel_diag.weight), dim=-1)
        if self.multi_c:
            c = self.c_layer.weight # (N_r, 1)

        # Dropout on edges
        num_edges = self.edge_index.size(1) // 2
        idx = torch.ones(num_edges)
        idx = self.edge_dropout(idx).bool()
        idx = idx.repeat(2)

        edge_index = self.edge_index[:, idx].to(x.device)
        edge_type = self.edge_type[idx].to(x.device)
        del idx

        for i, conv in enumerate(self.layers):
            x, r, c = conv.forward(x, edge_index, edge_type, (r, c))
            if i < len(self.layers) - 1:
                # Activation on r, which is not in the layer code
                r = tanh(r)
            assert not torch.isnan(x).any(), f"x contains nan values at layer {i+1}."
        del edge_index
        del edge_type

        c = F.softplus(c)
        if not self.multi_c:
            c = c.mean(dim=0, keepdim=True)
        return (x, r, c)
    

    def forward(self, queries, tails=None):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation)
            tails: torch.LongTensor with tails
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        while queries.dim() < 3:
            queries = queries.unsqueeze(1)
        if tails is not None:
            while tails.dim() < 2:
                tails = tails.unsqueeze(0)
        cache = self.forward_base()
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries, cache=cache)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(tails, cache=cache)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        # if tails is None: # Eval mode
        #     del cache[0], cache[1]
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))

        # get factors for regularization
        factors = self.get_factors(queries, tails)
        return predictions, factors
    
    def get_queries(self, queries, cache=None):
        if cache is None:
            x, r, curvatures = self.forward_base()
        else:
            x, r, curvatures = cache
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
    
    def get_rhs(self, tails=None, cache=None):
        if cache is None:
            x, r, curvatures = self.forward_base()
        else:
            x, r, curvatures = cache
        if tails is None:
            rhs_e, rhs_biases = x, self.bt.weight
            while rhs_e.dim() < 3:
                rhs_e = rhs_e.unsqueeze(0)
            while rhs_biases.dim() < 3:
                rhs_biases = rhs_biases.unsqueeze(0)
        else:
            rhs_e, rhs_biases = multi_index_select(x, tails), self.bt(tails)
            while rhs_e.dim() < 3:
                rhs_e = rhs_e.unsqueeze(1)
            while rhs_biases.dim() < 3:
                rhs_biases = rhs_biases.unsqueeze(1)
        return rhs_e, rhs_biases

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        rhs_e = expmap0_lorentz(rhs_e, c)
        dist = hyp_distance_multi_c_lorentz(lhs_e, rhs_e, c)
        return - dist ** 2