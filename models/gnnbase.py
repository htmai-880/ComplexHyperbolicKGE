import torch
from torch import nn
from models.base import KGModel
from datasets.kg_dataset import KGDataset
from utils.euclidean import multi_index_select
import numpy as np
import gc

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
    
    def get_x(self, x = None):
        if x is None:
            return self.entity.weight
        # Then x contains a N x 1 vector with the indices of the entities to fetch
        return self.entity(x.view(-1))
    
    def get_r(self):
        # For the user to write. For instance, for hyperbolic methods, we also need to include the curvature in the output.
        return torch.cat((self.rel.weight, self.rel_diag.weight), dim=-1)

    def forward_base(self, x = None, edge_index = None, edge_type = None, r = None):
        # x = tanh(self.entity.weight) # Typically, embeddings from language models will be reduced with an activation.
        x = self.get_x(x)
        r = self.get_r()

        # Dropout on edges
        if edge_index is None or edge_type is None:
            mask = self.edge_type < self.rel.weight.size(0) // 2
            edge_index = self.edge_index[:, mask].to(x.device)
            edge_type = self.edge_type[mask].to(x.device)
            idx = torch.ones_like(edge_type).float()
            idx = self.edge_dropout(idx).bool()
            edge_type = edge_type[idx]
            edge_index = edge_index[:, idx]

            # Add inverse edges
            edge_index_inv = torch.stack((edge_index[1], edge_index[0]), dim=0)
            edge_type_inv = edge_type + self.rel.weight.size(0) // 2
            edge_index = torch.cat((edge_index, edge_index_inv), dim=-1)
            edge_type = torch.cat((edge_type, edge_type_inv), dim=0)
            del idx, mask, edge_index_inv, edge_type_inv
        else:
            mask = edge_type < self.rel.weight.size(0) // 2
            edge_index = edge_index[:, mask].to(x.device)
            edge_type = edge_type[mask].to(x.device)
            idx = torch.ones_like(edge_type).float()
            idx = self.edge_dropout(idx).bool()
            edge_type = edge_type[idx]
            edge_index = edge_index[:, idx]

            # Add inverse edges
            edge_index_inv = torch.stack((edge_index[1], edge_index[0]), dim=0)
            edge_type_inv = edge_type + self.rel.weight.size(0) // 2
            edge_index = torch.cat((edge_index, edge_index_inv), dim=-1)
            edge_type = torch.cat((edge_type, edge_type_inv), dim=0)
            del idx, mask, edge_index_inv, edge_type_inv
            

        x, r = self.base(x, edge_index, edge_type, r)

        del edge_index
        del edge_type
        # Do NOT garbage collect. This makes training significantly slower.
        return x, r
    
    def forward(self, queries, tails=None, x = None, edge_index=None, edge_type=None):
        """KGModel forward pass. For GNNs, we will cache the embeddings and relations returned by the forward_base method.

        Args:
            queries: torch.LongTensor with query triples (head, relation)
            tails: torch.LongTensor with tails
            g: Data object with the graph
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
        cache = self.forward_base(x=x, edge_index=edge_index, edge_type=edge_type)
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries, cache=cache)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(tails, cache=cache, tails_idx=x.squeeze(-1) if not x is None else None)
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

    def get_rhs(self, tails=None, cache=None, tails_idx=None):
        if cache is None:
            x, r = self.forward_base()
        else:
            x, r = cache
        if tails is None:
            # The input graph might be just a subgraph of the full graph, so we need to select the
            # embeddings of the entities in the subgraph.
            rhs_e, rhs_biases = x, (self.bt.weight if tails_idx is None else self.bt(tails_idx))
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
        # return (torch.tensor([0.], device=queries.device, dtype=self.data_type),)
        return self.base.get_regularizable_params()
    
    def get_ranking(self, queries, filters, batch_size=500, chunk_size=1000, cache=None):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        chunk_size = chunk_size if chunk_size > 0 else self.sizes[2]
        ranks = torch.ones(len(queries), 1)
        device = self.entity.weight.device
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                # candidates = self.get_rhs(None, cache=cache) # (1, n_entities, d)
                c_top = min(c_begin + chunk_size, self.sizes[2])
                idx_chunk = torch.arange(c_begin, c_top).to(device)
                candidates = list(self.get_rhs(idx_chunk, cache=cache)) # (N,1,d)
                del idx_chunk
                candidates[0] = candidates[0].transpose(0, 1)
                candidates[1] = candidates[1].transpose(0, 1)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    # mask = torch.zeros((these_queries.size(0), self.entity.weight.size(0)), dtype=torch.bool)
                    # for i, query in enumerate(these_queries.numpy()):
                    #     mask[i, query[2]] = 1
                    #     mask[i, filters[tuple(query[:2])]] = 1
                    # mask = mask.to(device)
                    these_queries = these_queries.to(device)

                    q = self.get_queries(these_queries[..., :2], cache=cache) # (batch_size, 1, d)
                    rhs = self.get_rhs(these_queries[..., 2], cache=cache) # (batch_size, 1, d)
                    scores = self.score(q, candidates) # (batch_size, 1, 1)
                    targets = self.score(q, rhs) # ???
                    # scores.masked_fill_(mask.unsqueeze(-1), -1e6)

                    assert not scores.isnan().any()
                    assert not targets.isnan().any()
                    
                    # set filtered and true scores to -1e6 to be ignored
                    # for i, query in enumerate(these_queries.cpu().numpy()):
                    these_queries = these_queries.cpu().numpy()
                    for i, query in enumerate(these_queries):
                        filter_out = filters[tuple(query[:2])]
                        filter_out += [query[2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(k - c_begin) for k in filter_out if c_begin <= k < c_top
                            ]
                            scores[i, filter_in_chunk] = -1e6
                        else:
                            scores[i, filter_out] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    del these_queries
                    del q
                    del rhs
                    del scores
                    del targets
                c_begin += chunk_size
        del candidates
        gc.collect()
        return ranks.squeeze(1)
    
    def compute_metrics(self, examples, filters, batch_size=10):
        """Compute ranking-based evaluation metrics.
        In comparison to before, this one caches the entity and relation embeddings from the last layer.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        # rhs
        if isinstance(examples, tuple):
            examples = examples[0]
        if isinstance(examples, np.ndarray):
            examples = torch.from_numpy(examples)
        cache = self.forward_base()
        # Cache to cpu
        q = examples
        ranks = self.get_ranking(q, filters["rhs"], batch_size=batch_size, cache=cache)
        mean_rank["rhs"] = torch.mean(ranks).item()
        mean_reciprocal_rank["rhs"] = torch.mean(1. / ranks).item()
        hits_at["rhs"] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))

        # lhs
        q = torch.stack([examples[..., 2], examples[..., 1] + self.sizes[1] // 2, examples[..., 0]], dim=-1)
        ranks = self.get_ranking(q, filters["lhs"], batch_size=batch_size, cache=cache)
        mean_rank["lhs"] = torch.mean(ranks).item()
        mean_reciprocal_rank["lhs"] = torch.mean(1. / ranks).item()
        hits_at["lhs"] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))
        
        return mean_rank, mean_reciprocal_rank, hits_at