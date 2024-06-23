# Inherited from https://github.com/HazyResearch/KGEmb
"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection, multi_bmm, givens_unitary, norm_clamp
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, logmap0

from utils.hyperbolic import lorentz_boost, expmap0_lorentz, hyp_distance_multi_c_lorentz, logmap0_lorentz

HYP_MODELS = ["RotH", "RefH", "AttH", "AttRH", "IFFTH", "IsoH", "RotLH", "HyboNet"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        """
        rank: dim
        entity: nn.Embedding, size = (n_entities, dim)
        rel: nn.Embedding, size = (n_relations, dim)
        rel_daig: nn.Embedding, size = (n_relations, dim), what is this???
        multi_c: bool
        c_init: tensor, size = dim
        """
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.multi_c = args.multi_c
        if self.multi_c:
            self.c = nn.Embedding(self.sizes[1], 1)
        else:
            self.c = nn.Embedding(1, 1)

        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.ones_(self.c.weight)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        rhs_e = expmap0(rhs_e, c)
        return - hyp_distance_multi_c(lhs_e, rhs_e, c) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = expmap0(self.entity(queries[..., 0]), c)   # hyperbolic
        rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        lhs = project(mobius_add(head, rel1, c), c)   # hyperbolic
        res1 = givens_rotations(self.rel_diag(queries[..., 1]), lhs)   # givens_rotation(Euclidean, hyperbolic)
        res2 = mobius_add(res1, rel2, c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        rel, _ = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel = expmap0(rel, c)   # hyperbolic
        lhs = givens_reflection(self.rel_diag(queries[..., 1]), self.entity(queries[..., 0]))   # givens_reflection(Euclidean, Euclidean)
        lhs = expmap0(lhs, c)   # hyperbolic
        res = project(mobius_add(lhs, rel, c), c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=-2)
        self.scale = 1. / np.sqrt(self.rank)

        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.normal_(self.context_vec.weight, 0, self.init_size)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = self.entity(queries[..., 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[..., 1]), 2, dim=-1)
        rot_q = givens_rotations(rot_mat, head).unsqueeze(-2)
        ref_q = givens_reflection(ref_mat, head).unsqueeze(-2)
        cands = torch.cat([ref_q, rot_q], dim=-2)
        context_vec = self.context_vec(queries[..., 1]).unsqueeze(-2)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=-2)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases


class AttRH(BaseH):
    def __init__(self, args):
        super(AttRH, self).__init__(args)
        del self.rel_diag
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.weights = nn.Embedding(self.sizes[1], 2)
        self.act = nn.Softmax(dim=-1)

        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.normal_(self.weights.weight, 0, self.init_size)
        
    def get_queries(self, queries):
        c = F.softplus(self.c(queries[..., 1]))
        head = expmap0(self.entity(queries[..., 0]), c)   # hyperbolic
        rel = self.rel(queries[..., 1])
        rel_diag = self.rel_diag(queries[..., 1])

        head_rot, head_ref = torch.chunk(head, 2, dim=-1)
        rel_rot, rel_ref = torch.chunk(rel, 2, dim=-1)
        rel_diag_rot, rel_diag_ref = torch.chunk(rel_diag, 2, dim=-1)
        
        # Rotation
        rel1, rel2 = torch.chunk(rel_rot, 2, dim=-1)   # Euclidean
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        lhs = project(mobius_add(head_rot, rel1, c), c)   # hyperbolic
        res_rot = givens_rotations(rel_diag_rot, lhs)   # givens_rotation(Euclidean, hyperbolic)
        res_rot = mobius_add(res_rot, rel2, c)   # hyperbolic


        # Reflection
        rel, _ = torch.chunk(rel_ref, 2, dim=-1)   # Euclidean
        rel = expmap0(rel, c)   # hyperbolic
        lhs = givens_reflection(rel_diag_ref, head_ref)   # givens_reflection(Euclidean, Euclidean)
        lhs = expmap0(lhs, c)   # hyperbolic
        res_ref = project(mobius_add(lhs, rel, c), c)   # hyperbolic

        res2 = torch.cat([res_rot, res_ref], dim=-1)
        lhs_biases = self.bh(queries[..., 0])

        # Get attention parameter
        weights = self.weights(queries[..., 1]).unsqueeze(-2)
        weights = self.act(weights) # (batch_size, ..., 2)

        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        while weights.dim() < 3:
            weights = weights.unsqueeze(1)
        return (res2, c, weights), lhs_biases

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c, weights = lhs_e
        lhs_rot, lhs_ref = torch.chunk(lhs_e, 2, dim=-1)
        rhs_rot, rhs_ref = torch.chunk(rhs_e, 2, dim=-1)
        return - weights[..., 0:1] * hyp_distance_multi_c(lhs_rot, rhs_rot, c) ** 2 - weights[..., 1:] * hyp_distance_multi_c(lhs_ref, rhs_ref, c) ** 2



class IsoH(BaseH):
    def __init__(self, args):
        super(IsoH, self).__init__(args)
        del self.rel_diag
        self.rel_diag = nn.Embedding(self.sizes[1], 2*self.rank)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            # Initialize scaling at 1
            nn.init.ones_(self.rel_diag.weight[..., self.rank:])
    
    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = expmap0(self.entity(queries[..., 0]), c)   # hyperbolic
        rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        lhs = project(mobius_add(head, rel1, c), c)   # hyperbolic
        r = self.rel_diag(queries[..., 1])
        rot, scale = r[..., :self.rank], r[..., self.rank:]
        scale1, scale2 = torch.chunk(scale, 2, dim=-1)
        lhs = logmap0(lhs, c)
        res1 = givens_rotations(rot, lhs, scale=scale1)   # givens_rotation(Euclidean, hyperbolic)
        res1[..., 0::2].mul_(scale2)
        res1[..., 1::2].mul_(scale2)
        res1 = expmap0(res1, c)
        res2 = project(mobius_add(res1, rel2, c), c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases



class IFFTH(BaseH):
    def __init__(self, args):
        # To parametrize the unitary transformations, we need d angles and d radii
        # if args.rank = N = 2(n-1), then we take the n-dimensional complex unit ball.
        # setting n = 2d, it will hold:
        # n = (N/2) + 1
        # To ensure everything goes smoothly, we should pick N even, but n also even.
        # Example: N = 34, n = 34/2 + 1 = 18

        super(IFFTH, self).__init__(args)
        n = (self.rank // 2) + 1 # = 2d, dimension of the complex unit ball
        d = n//2
        assert n == 2*d, f"n = {n} is not of even dimension (rank = {self.rank})."
        self.rel_diag = nn.Embedding(self.sizes[1], 3*n)
        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

    def get_queries(self, queries):
        c = F.softplus(self.c(queries[..., 1])) if self.multi_c else self.c.weight
        head = self.entity(queries[..., 0]) # euclidean
        head = expmap0(self.entity(queries[..., 0]), c)   # hyperbolic
        rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        head = project(mobius_add(head, rel1, c), c)   # hyperbolic
        # head = logmap0(rel1, c) # euclidean

        # FFT
        head_f = torch.fft.rfft(head, norm="ortho") # complex (batch_size, ..., 2d)
        rel_diag = self.rel_diag(queries[..., 1])
        a, b, angle = torch.chunk(rel_diag, 3, dim=-1)

        # Unitary transform
        head_f = givens_unitary(a, b, angle, head_f)

        # IFFT
        head = torch.fft.irfft(head_f, norm="ortho") # Euclidean
        # head = expmap0(head, c) # hyperbolic

        # Add
        res2 = project(mobius_add(head, rel2, c), c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases










        

###### Hyperboloid models

class BaseLorentz(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        """
        rank: dim
        entity: nn.Embedding, size = (n_entities, dim)
        rel: nn.Embedding, size = (n_relations, dim)
        rel_daig: nn.Embedding, size = (n_relations, dim), what is this???
        multi_c: bool
        c_init: tensor, size = dim
        """
        super(BaseLorentz, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.multi_c = args.multi_c
        if self.multi_c:
            self.c = nn.Embedding(self.sizes[1], 1)
        else:
            self.c = nn.Embedding(1, 1)

        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.ones_(self.c.weight)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        rhs_e = expmap0_lorentz(rhs_e, c)
        return - hyp_distance_multi_c_lorentz(lhs_e, rhs_e, c) ** 2
    
class RotLH(BaseLorentz):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args):
        super(RotLH, self).__init__(args)
        del self.rel_diag
        self.rel_diag = nn.Embedding(self.sizes[1], 2*(self.rank))
        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            # Initialize scaling at 0
            nn.init.ones_(self.rel_diag.weight[..., self.rank:])

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = expmap0_lorentz(self.entity(queries[..., 0]), c)   # hyperbolic
        rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean velocities
        lhs = lorentz_boost(head, rel1, c)   # hyperbolic
        # lhs = head
        # res1 = givens_rotations(self.rel_diag(queries[..., 1]), lhs, scale=scale)   # givens_rotation(Euclidean, hyperbolic)
        r = self.rel_diag(queries[..., 1])
        rot, scale = r[..., :self.rank], r[..., self.rank:]
        scale1, scale2 = torch.chunk(scale, 2, dim=-1)
        lhs = logmap0_lorentz(lhs, c)
        res1 = givens_rotations(rot, lhs, scale=scale1)   # givens_rotation(Euclidean, hyperbolic)
        res1[..., 0::2].mul_(scale2)
        res1[..., 1::2].mul_(scale2)
        res1 = expmap0_lorentz(res1, c)
        res2 = lorentz_boost(res1, rel2, c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases

class HyboNet(BaseLorentz):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args):
        super(HyboNet, self).__init__(args)
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], (self.rank+1)**2)
        del self.rel_diag
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank+2)
        with torch.no_grad():
            nn.init.normal_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.ones_(self.rel_diag.weight[..., -1])
            # Initialize scaling at 0
    
    def lorentz_linear(self, x, weight, scale, bias=None, c = None):
        x = multi_bmm(x.unsqueeze(-2), weight.transpose(-2, -1)).squeeze(-2)
        epsilon = 1.1 if c is None else (1/c**0.5) + 0.1
        time = x.narrow(-1, 0, 1).sigmoid() * scale + epsilon
        if bias is not None:
            x = x + bias
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        x_narrow = x_narrow / ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) / (time * time - 1)).sqrt()
        # x = torch.cat([time, x_narrow], dim=-1)
        return x_narrow

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = expmap0_lorentz(self.entity(queries[..., 0]), c)   # hyperbolic
        # Make explicit
        head0 = torch.sum(head**2, dim=-1, keepdim=True) + 1/c
        head0 = torch.sqrt(head0)
        head = torch.cat([head0, head], dim=-1)

        rel_transform, rel = self.rel(queries[..., 1]), self.rel_diag(queries[..., 1])
        rel_bias, rel_scale = rel[..., :-1], rel[..., -1:]
        rel_scale = torch.abs(rel_scale)
        rel_transform = rel_transform.view(*rel_transform.shape[:-1], self.rank+1, self.rank+1)

        # Compute transform
        res2 = self.lorentz_linear(head, rel_transform, rel_scale, rel_bias, c)

        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases