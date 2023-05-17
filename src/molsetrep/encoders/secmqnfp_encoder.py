from typing import Iterable, Any, Optional

import torch
import numpy as np

from torch.utils.data import TensorDataset
from mhfp.encoder import MHFPEncoder
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem import (
    rdMolDescriptors,
    MolFromSmiles,
    GetSymmSSSR,
    Descriptors,
)

from molsetrep.encoders.encoder import Encoder


class SECMQNFPEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("SECMQNFPEncoder")

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps = []
        for smi in smiles:
            substructs = set(
                MHFPEncoder.shingling_from_mol(
                    MolFromSmiles(smi), radius, rings, kekulize, min_radius
                )
            )

            fp = []

            for substruct in substructs:
                submol = MolFromSmiles(substruct, sanitize=False)
                submol.UpdatePropertyCache(strict=False)
                GetSymmSSSR(submol)
                ds = rdMolDescriptors.MQNs_(submol)
                ds = list(ds)
                ds.append(Descriptors.MolLogP(submol))
                ds.append(Descriptors.TPSA(submol))
                fp.append(np.array(ds))

            fps.append(fp)

        return super().to_tensor_dataset(fps, labels, label_dtype)
