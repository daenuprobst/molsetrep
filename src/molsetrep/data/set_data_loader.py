from typing import Iterable, Optional, Sequence, Union

import torch

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


# class SetDataLoader(DataLoader):
#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: Optional[int] = 1,
#         shuffle: Optional[bool] = None,
#         sampler: Optional[Union[Sampler, Iterable]] = None,
#         batch_sampler: Optional[Union[Sampler[Sequence], Iterable[Sequence]]] = None,
#         num_workers: int = 0,
#         collate_fn: Optional[_collate_fn_t] = None,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         timeout: float = 0,
#         worker_init_fn: Optional[_worker_init_fn_t] = None,
#         multiprocessing_context=None,
#         generator=None,
#         *,
#         prefetch_factor: Optional[int] = None,
#         persistent_workers: bool = False,
#         pin_memory_device: str = ""
#     ):
#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             sampler,
#             batch_sampler,
#             num_workers,
#             collate_fn,
#             pin_memory,
#             drop_last,
#             timeout,
#             worker_init_fn,
#             multiprocessing_context,
#             generator,
#             prefetch_factor=prefetch_factor,
#             persistent_workers=persistent_workers,
#             pin_memory_device=pin_memory_device,
#         )
