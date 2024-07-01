from typing import List, Literal, Optional, Tuple, Union, overload
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils.map import map_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import select
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.typing import OptTensor



def subgraph2(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    exclude : Optional[Tensor] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = torch.tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """
    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset
        subset = node_mask.nonzero().view(-1)
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    if not exclude is None:
        edge_mask = edge_mask & (~exclude)
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index, _ = map_index(
            edge_index.view(-1),
            subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index.view(2, -1)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


import copy

def make_subgraph(g, subset, exclude=None, account_for_train_mask=True) -> Data:
    r"""Returns the induced subgraph given by the node indices
    :obj:`subset`.

    Args:
        subset (LongTensor or BoolTensor): The nodes to keep.
    """
    if not exclude is None:
        if account_for_train_mask:
            # exclude contains batch.input_id, which is only valid with the train_mask
            exclude_ = torch.zeros(g.train_mask.sum(), dtype=torch.bool, device=g.edge_index.device)
            exclude_[exclude] = True
            exclude = torch.zeros(g.num_edges, dtype=torch.bool, device=exclude_.device)
            exclude[g.train_mask] = exclude_
        else:
            exclude_ = torch.zeros(g.num_edges, dtype=torch.bool)
            exclude_[exclude] = True
            exclude = exclude_
    if 'edge_index' in g:
        edge_index, _, edge_mask = subgraph2(
            subset,
            g.edge_index,
            relabel_nodes=True,
            num_nodes=g.num_nodes,
            exclude=exclude,
            return_edge_mask=True,
        )
    else:
        edge_index = None
        edge_mask = torch.ones(
            g.num_edges,
            dtype=torch.bool,
            device=subset.device,
        )

    data = copy.copy(g)

    for key, value in g:
        if key == 'edge_index':
            data.edge_index = edge_index
        elif key == 'num_nodes':
            if subset.dtype == torch.bool:
                data.num_nodes = int(subset.sum())
            else:
                data.num_nodes = subset.size(0)
        elif g.is_node_attr(key):
            cat_dim = g.__cat_dim__(key, value)
            data[key] = select(value, subset, dim=cat_dim)
        elif g.is_edge_attr(key):
            cat_dim = g.__cat_dim__(key, value)
            data[key] = select(value, edge_mask, dim=cat_dim)

    return data, edge_mask