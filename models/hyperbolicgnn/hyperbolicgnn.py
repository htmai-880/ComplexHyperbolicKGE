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
from models.mlp import MLP




class HyperbolicBase(BaseGNN):
    def __init__(self, **kwargs):
        super(HyperbolicBase, self).__init__(**kwargs)
        self.act_r_base = kwargs.get("act_r")
        self.act_r = (lambda x : (self.act_r_base(x[0]), x[1]))