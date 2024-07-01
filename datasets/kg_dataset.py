"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from .sparse import SparseDataset, sparse_batch_collate
from utils.pyg_utils import make_subgraph
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils.map import map_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
    
class KGDataset2(KGDataset):
    def __init__(self, data_path, debug):
        super(KGDataset2, self).__init__(data_path, debug)

        # Get train examples
        train_examples = self.data["train"]
        # Make a specific train filter
        self.train_filter = self.make_train_filter(train_examples, None)
        valid_examples = self.data["valid"]
        # Make a specific valid filter
        self.valid_filter = self.make_train_filter(valid_examples, self.train_filter)
    
    def make_train_filter(self, examples, other_filter=None):
        # Train filter first
        filter = dict() if other_filter is None else copy.deepcopy(other_filter)
        if not other_filter is None:
            # Turn lists to sets
            for key in filter:
                filter[key] = set(filter[key])

        n_relations = self.n_predicates // 2
        for sub, rel, obj in examples:
            if (sub, rel) not in filter:
                filter[(sub, rel)] = set()
            filter[(sub, rel)].add(obj)
            if (obj, rel + n_relations) not in filter:
                filter[(obj, rel + n_relations)] = set()
            filter[(obj, rel + n_relations)].add(sub)
        # Turn sets to lists
        for key in filter:
            filter[key] = list(filter[key])
        return filter
    
    def make_labels(self, examples, filter):
        # Make labels as sparse tensor
        row, col = [], []
        for i, (sub, rel, obj) in enumerate(examples):
            label = filter[(sub, rel)]
            for l in label:
                row.append(i)
                col.append(l)
        labels = sp.csr_matrix(
            (np.ones(len(row)), (row, col)), shape=(examples.shape[0], self.n_entities)
        )
        return labels
    
    def get_examples(self, split, rel_idx=-1):
        examples = super().get_examples(split, rel_idx).numpy()
        if split == "test":
            return examples, None
        filter = {
            "train": self.train_filter,
            "valid": self.valid_filter
        }[split]
        labels = self.make_labels(examples, filter)
        return examples, labels
    


class KGDataset3(KGDataset):
    def __init__(self, data_path, debug):
        super(KGDataset3, self).__init__(data_path, debug)
        # Create PyG Data from it
        train_examples = self.data["train"]
        # add inverse triples
        copy = np.copy(train_examples)
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2
        train_examples = np.copy(np.vstack((train_examples, copy)))

        valid_triples = self.data["valid"]
        # add inverse triples
        copy = np.copy(valid_triples)
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2
        valid_triples = np.copy(np.vstack((valid_triples, copy)))

        # Stack all triples
        all_triples = np.vstack((train_examples, valid_triples))
        all_triples = torch.from_numpy(all_triples.astype("int64"))
        train_mask = torch.zeros(all_triples.size(0), dtype=torch.bool)
        train_mask[:train_examples.shape[0]] = True
        self.g_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.g = Data(
            x = torch.arange(self.n_entities).unsqueeze(-1),
            edge_index=all_triples[:, [0, 2]].t().contiguous(),
            edge_type=all_triples[:, 1],
            train_mask=train_mask,
            val_mask=~train_mask,
            num_nodes=self.n_entities
        ).to(self.g_device)
        print("PyG Graph: ", self.g)

    def make_loader(self, batch_size=4, shuffle=True, num_workers=-1, split="train"):
        return LinkNeighborLoader(
            self.g,
            num_neighbors=[20,20],
            batch_size=batch_size,
            edge_label_index=self.g.edge_index[:, self.g.train_mask] if split=="train" else self.g.edge_index,
            edge_label = self.g.edge_type[self.g.train_mask] if split=="train" else self.g.edge_type,
            shuffle=shuffle,
            num_workers=num_workers
        )
    def _make_labels(self, edge_type, edge_index, num_nodes, queries=None):
        q = edge_index[0] * self.n_predicates + edge_type
        targets = edge_index[1]
        ind = torch.stack([q, targets]).to(q.device)
        if queries is None:
            labels = torch.sparse_coo_tensor(
                indices=ind,
                values=torch.ones_like(q),
                size=(self.n_predicates * num_nodes, num_nodes),
                device=q.device
            )
            return torch.index_select(labels, 0, q)
        # Otherwise, hash the queries as well
        queries = queries.to(q.device)
        queries_hash = queries[:, 0] * self.n_predicates + queries[:, 1]
        ind_new = torch.stack([queries_hash, queries[:, 2]]).to(q.device)
        ind = torch.cat([ind, ind_new], dim=1)
        labels = torch.sparse_coo_tensor(
            indices=ind,
            values=torch.ones_like(ind[0]),
            size=(self.n_predicates * num_nodes, num_nodes),
            device=q.device
        )
        return torch.index_select(labels, 0, queries_hash)

    
    def make_labels(self, subgraph_g, split="train", triples=None):
        """Make a B x N sparse tensor containing the labels for each edge in the subgraph. B is the number of queries and N is the number of nodes in the subgraph.

        Args:
            subgraph_g (Data): The subgraph containing the edges and nodes.
            n_predicates (int): The number of predicates in the entire graph.
        
        Returns:
            torch.sparse.FloatTensor: The labels for each edge in the subgraph. output[i, j] = 1 if the j-th node is the target of the i-th query.
        """
        if split == "train":
            edge_index = subgraph_g.edge_index[:, subgraph_g.train_mask]
            edge_type = subgraph_g.edge_type[subgraph_g.train_mask]
        elif split == "val":
            edge_index = subgraph_g.edge_index[:, subgraph_g.val_mask | subgraph_g.train_mask]
            edge_type = subgraph_g.edge_type[subgraph_g.val_mask | subgraph_g.train_mask]
        else:
            edge_index = subgraph_g.edge_index
            edge_type = subgraph_g.edge_type
        return self._make_labels(edge_type, edge_index, num_nodes=subgraph_g.num_nodes, queries=triples)


    def make_subgraph(self, batch, split="train", return_labels=False):
        # Retrieve the triples
        src, dist, e_type = batch.n_id[batch.edge_label_index[0]], batch.n_id[batch.edge_label_index[1]], batch.edge_label
        batch_triples = torch.stack([src, dist], dim=1)
        num_nodes = maybe_num_nodes(self.g.edge_index, num_nodes=self.g.num_nodes)
        batch_triples, _ = map_index(
            batch_triples.view(-1),
            index=batch.n_id,
            max_index=num_nodes,
            inclusive=True
        )
        batch_triples = batch_triples.view(2, -1)
        batch_triples = torch.stack([
            batch_triples[0], batch.edge_label, batch_triples[1]
        ], dim=1)
        subgraph_g, _ = make_subgraph(self.g, batch.n_id, exclude=batch.input_id, account_for_train_mask=True)
        # Note: this subgraph is more complete than the one in input
        if not return_labels:
            return (subgraph_g, batch_triples)
        labels = self.make_labels(subgraph_g, split=split, triples=batch_triples)
        return subgraph_g, batch_triples, labels
    
    def get_triples(self, subgraph, split="train"):
        edge_index = subgraph.edge_index[:, subgraph.train_mask] if split == "train" else subgraph.edge_index
        edge_type = subgraph.edge_type[subgraph.train_mask] if split == "train" else subgraph.edge_type
        return torch.stack([edge_index[0], edge_type, edge_index[1]], dim=1)


        

