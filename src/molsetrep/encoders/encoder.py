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
        Xn: Iterable[Iterable[Iterable[np.ndarray]]],
        y: Iterable[Any],
        y_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        Xn_tensors = []
        for X in Xn:
            n = len(X)
            d = -1

            for x in X:
                if len(x) > 0:
                    d = len(x[0])
                    break

            max_cardinality = max([len(x) for x in X])

            X_tensor = torch.zeros((n, max_cardinality, d))

            for i, x in enumerate(X):
                X_tensor[i, : len(x), :] = torch.FloatTensor(
                    np.nan_to_num(
                        np.array(x), nan=0.0, posinf=10**6, neginf=-(10**6)
                    )
                )

            Xn_tensors.append(X_tensor)

        return TensorDataset(*Xn_tensors, torch.tensor(y, dtype=y_dtype))

    def encode(self, **kwargs) -> TensorDataset:
        ...
