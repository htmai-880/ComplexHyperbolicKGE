from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (random, 
                          coo_matrix,
                          csr_matrix, 
                          vstack)
from tqdm import tqdm

class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                 targets:Union[np.ndarray, coo_matrix, csr_matrix], 
                 transform:bool = None):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.examples = data.tocsr()
        else:
            self.examples = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets
        
        self.transform = transform # Can be removed

    def __getitem__(self, index):
        return self.examples[index], self.targets[index]

    def __len__(self):
        return self.examples.shape[0]
      
def sparse_coo_to_tensor2(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col) # np.vstack
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    s = torch.Size(shape)

    return torch.sparse.DoubleTensor(i, v, s)
    
def sparse_batch_collate(batch): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]
    
    if type(data_batch[0]) == csr_matrix:
        data_batch = data_batch.tocoo() # removed vstack
        data_batch = sparse_coo_to_tensor2(data_batch)
    else:
        data_batch = torch.LongTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo() # removed vstack
        targets_batch = sparse_coo_to_tensor2(targets_batch)
    else:
        targets_batch = torch.LongTensor(targets_batch)
    return data_batch, targets_batch


