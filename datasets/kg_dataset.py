"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from .sparse import SparseDataset, sparse_batch_collate
from torch.utils.data.sampler import BatchSampler, RandomSampler

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


class TrainDataset(SparseDataset):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug, dtype=None):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}

        if dtype == None:
            self.data_type = torch.float
        elif dtype == "double":
            self.data_type = torch.double
        elif dtype == "float":
            self.data_type = torch.float

        split = "train"
        file_path = os.path.join(self.data_path, split + ".pickle")
        with open(file_path, "rb") as in_file:
            self.data = pkl.load(in_file)

        max_axis = np.max(self.data, axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

        self.examples = self.get_examples()
        # Make a specific train filter
        self.train_filter = self.make_filters()
        self.targets = self.make_labels()

        super(TrainDataset, self).__init__(self.examples, self.targets, transform=False)

    
    def make_filters(self):
        # Train filter first
        train_filter = dict()
        # rhs filter
        train_filter["rhs"] = dict()
        train_filter["lhs"] = dict()
        for sub, rel, obj in self.examples:
            if (sub, rel) not in train_filter["rhs"]:
                train_filter["rhs"][(sub, rel)] = set()
            train_filter["rhs"][(sub, rel)].add(obj)
            if (rel, obj) not in train_filter["lhs"]:
                train_filter["lhs"][(rel, obj)] = set()
            train_filter["lhs"][(rel, obj)].add(sub)

        # Turn sets to lists
        for key in train_filter["rhs"]:
            train_filter["rhs"][key] = list(train_filter["rhs"][key])
        for key in train_filter["lhs"]:
            train_filter["lhs"][key] = list(train_filter["lhs"][key])

        return train_filter
    
    def make_labels(self):
        # Make labels as sparse tensor
        row, col = [], []
        filters = self.train_filter
        for i, (sub, rel, obj) in enumerate(self.examples):
            label = filters["rhs"][(sub, rel)]
            for l in label:
                row.append(i)
                col.append(l)
        labels = sp.csr_matrix(
            (np.ones(len(row)), (row, col)), shape=(len(row), self.n_entities)
        )
        return labels

    def get_examples(self, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data
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
        return examples.astype("int64")

    def get_filters(self, split=None):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.train_filter

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
    

class ValidDataset(SparseDataset):
    """Knowledge Graph dataset class."""
    def __init__(self, train_dataset):
        # Initialize from the train dataset
        self.data_path = train_dataset.data_path
        self.debug = train_dataset.debug
        self.data_type = train_dataset.data_type


        self.data = {}
        split = "valid"
        file_path = os.path.join(self.data_path, split + ".pickle")
        with open(file_path, "rb") as in_file:
            self.data = pkl.load(in_file)

        self.n_entities = train_dataset.n_entities
        self.n_predicates = train_dataset.n_predicates
    
        self.examples = self.get_examples()
        self.valid_filter = self.make_filters(train_dataset.train_filter)
        self.targets = self.make_labels()
        self.examples = torch.from_numpy(self.examples)

        super(ValidDataset, self).__init__(self.examples, self.targets, transform=False)
        
    
    def make_filters(self, train_filter):
        # deep copy of train filter
        valid_filter = copy.deepcopy(train_filter)
        # To set:
        for key in valid_filter["rhs"]:
            valid_filter["rhs"][key] = set(valid_filter["rhs"][key])
        for key in valid_filter["lhs"]:
            valid_filter["lhs"][key] = set(valid_filter["lhs"][key])

        # Valid filter from valid examples
        for sub, rel, obj in self.examples:
            if (sub, rel) not in valid_filter["rhs"]:
                valid_filter["rhs"][(sub, rel)] = set()
            valid_filter["rhs"][(sub, rel)].add(obj)
            if (rel, obj) not in valid_filter["lhs"]:
                valid_filter["lhs"][(rel, obj)] = set()
            valid_filter["lhs"][(rel, obj)].add(sub)
        
        # Turn sets to lists
        for key in valid_filter["rhs"]:
            valid_filter["rhs"][key] = list(valid_filter["rhs"][key])
        for key in valid_filter["lhs"]:
            valid_filter["lhs"][key] = list(valid_filter["lhs"][key])
        return valid_filter

    def get_examples(self, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return examples.astype("int64")
    
    def get_label(self, label):
        y = np.zeros([self.n_entities])
        y[label] = 1
        return torch.from_numpy(y, dtype=self.data_type)
    
    def make_labels(self):
        # Make labels as sparse tensor
        row, col = [], []
        filters = self.valid_filter
        for i, (sub, rel, obj) in enumerate(self.examples):
            label = filters["rhs"][(sub, rel)]
            for l in label:
                row.append(i)
                col.append(l)
        labels = sp.csr_matrix(
            (np.ones(len(row)), (row, col)), shape=(len(row), self.n_entities)
        )
        return labels
    
    def get_filters(self, split=None):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.valid_filter
    
    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
    

class KGDataset2(object):
    def __init__(self, data_path, debug, batch_size = 100, dtype=None):
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["test"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
            
        self.train_dataset = TrainDataset(data_path, debug, dtype)
        self.valid_dataset = ValidDataset(self.train_dataset)
        self.train_loader = self.make_loader("train", batch_size)
        self.valid_loader = self.make_loader("valid", batch_size)

        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()

        self.n_entities = self.train_dataset.n_entities
        self.n_predicates = self.train_dataset.n_predicates

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
    
    def make_loader(self, split="train", batch_size=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if split == "train":
            sampler = BatchSampler(
                RandomSampler(self.train_dataset, generator = torch.Generator(device=device)),
                batch_size=batch_size,
                drop_last=False,
            )
            return DataLoader(self.train_dataset, batch_size=1, collate_fn=sparse_batch_collate,
                              generator=torch.Generator(device=device), sampler=sampler
            )
        elif split == "valid":
            sampler = BatchSampler(
                RandomSampler(self.valid_dataset, generator = torch.Generator(device=device)),
                batch_size=batch_size,
                drop_last=False,
            )
            return DataLoader(self.valid_dataset, batch_size=1, collate_fn=sparse_batch_collate,
                              generator=torch.Generator(device=device), sampler=sampler
            )
    
    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        if split=="train":
            examples = self.train_dataset.get_examples(rel_idx)
        elif split=="valid":
            examples = self.valid_dataset.get_examples(rel_idx)
        else:
            examples = self.data[split]
            if rel_idx >= 0:
                examples = examples[examples[:, 1] == rel_idx]
            if self.debug:
                examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))
