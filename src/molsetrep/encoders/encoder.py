from typing import Any, Iterable, Optional

import torch
import numpy as np

from torch.utils.data import TensorDataset
from rdkit.Chem.rdchem import Mol


class Encoder:
    def __init__(self, name) -> "Encoder":
        self.__encoder_name__ = name

    def to_tensor_dataset(
        self,
        X: Iterable[Iterable[np.ndarray]],
        y: Iterable[Any],
        y_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        n = len(X)
        d = len(X[0][0])
        max_cardinality = max([len(x) for x in X])

        X_tensor = torch.zeros((n, max_cardinality, d))

        for i, x in enumerate(X):
            X_tensor[i, : len(x), :] = torch.FloatTensor(np.array(x))

        return TensorDataset(X_tensor, torch.tensor(y, dtype=y_dtype))

    def to_multi_tensor_dataset(
        self,
        X: Iterable[Iterable[np.ndarray]],
        X2: Iterable[Iterable[np.ndarray]],
        y: Iterable[Any],
        y_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        n = len(X)
        d = len(X[0][0])
        max_cardinality = max([len(x) for x in X])

        X_tensor = torch.zeros((n, max_cardinality, d))

        for i, x in enumerate(X):
            X_tensor[i, : len(x), :] = torch.FloatTensor(np.nan_to_num(np.array(x)))

        n = len(X2)
        d = len(X2[0][0])
        max_cardinality = max([len(x) for x in X2])

        X2_tensor = torch.zeros((n, max_cardinality, d))

        for i, x in enumerate(X2):
            X2_tensor[i, : len(x), :] = torch.FloatTensor(np.nan_to_num(np.array(x)))

        return TensorDataset(X_tensor, X2_tensor, torch.tensor(y, dtype=y_dtype))

    def encode(
        self,
    ) -> TensorDataset:
        ...
