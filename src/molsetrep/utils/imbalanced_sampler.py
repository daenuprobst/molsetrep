from typing import Optional

import torch
from torch import Tensor

from torch.utils.data import Dataset


class ImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    r"""A weighted random sampler that randomly samples elements according to
    class distribution.
    As such, it will either remove samples from the majority class
    (under-sampling) or add more examples from the minority class
    (over-sampling).

    **Graph-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, ImbalancedSampler

        sampler = ImbalancedSampler(dataset)
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)

    **Node-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    You can also pass in the class labels directly as a :class:`torch.Tensor`:

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data.y)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    Args:
        dataset (Dataset): The dataset or class distribution
            from which to sample the data, given as a
            :class:`~torch.utils.data.Dataset` object.
        num_samples (int, optional): The number of samples to draw for a single
            epoch. If set to :obj:`None`, will sample as much elements as there
            exists in the underlying data. (default: :obj:`None`)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
    ):
        ys = [data[-1] for data in dataset]
        y = torch.tensor(ys).view(-1)
        assert len(dataset) == y.numel()

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1.0 / y.bincount()
        weight = class_weight[y]

        return super().__init__(weight, num_samples, replacement=True)
