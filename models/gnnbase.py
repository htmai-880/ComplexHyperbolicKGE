import torch
from torch import nn
from models.base import KGModel
from datasets.kg_dataset import KGDataset
from utils.euclidean import multi_index_select

class GNN(KGModel):
    def __init__(self, args, dataset: KGDataset):
        super(GNN, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.edge_dropout = nn.Dropout(args.edge_dropout)
                # Fetch the triples from the dataset
        train_examples = dataset.get_examples("train")
        if isinstance(train_examples, tuple):
            train_examples = train_examples[0]
        if not isinstance(train_examples, torch.Tensor):
            train_examples = torch.from_numpy(train_examples)
        self.edge_index = train_examples[:, [0, 2]].t().contiguous()
        self.edge_type = train_examples[:, 1].contiguous()
        self.hidden_dim = args.hidden_dim if args.hidden_dim else self.rank
        self.base = None # Changed by the user in function of the model
    
    def __setattr__(self, name, value):
        if not "edge_" in name:
            super().__setattr__(name, value)
        else:
            nn.Module.__setattr__(self, name, value)
    
    def get_x(self):
        return self.entity.weight
    
    def get_r(self):
        # For the user to write. For instance, for hyperbolic methods, we also need to include the curvature in the output.
        return torch.cat((self.rel.weight, self.rel_diag.weight), dim=-1)

    def forward_base(self):
        # x = tanh(self.entity.weight) # Typically, embeddings from language models will be reduced with an activation.
        x = self.get_x()
        r = self.get_r()

        # Dropout on edges
        num_edges = self.edge_index.size(1) // 2
        idx = torch.ones(num_edges)
        idx = self.edge_dropout(idx).bool()
        idx = idx.repeat(2)

        edge_index = self.edge_index[:, idx].to(x.device)
        edge_type = self.edge_type[idx].to(x.device)
        del idx

        x, r = self.base(x, edge_index, edge_type, r)

        del edge_index
        del edge_type

        return x, r
    
    def forward(self, queries, tails=None):
        """KGModel forward pass. For GNNs, we will cache the embeddings and relations returned by the forward_base method.

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
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))

        # get factors for regularization
        factors = self.get_factors(queries, tails)
        return predictions, factors
    
    def get_queries(self, queries, cache=None):
        # For the user to write
        if cache is None:
            x, r = self.forward_base()
        else:
            x, r = cache
        pass

    def get_rhs(self, tails=None, cache=None):
        if cache is None:
            x, r = self.forward_base()
        else:
            x, r = cache
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

    def get_factors(self, queries, tails=None):
        return self.base.get_regularizable_params()