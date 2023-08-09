from typing import Iterable, Any, Optional

import torch
import numpy as np

from torch.utils.data import TensorDataset
from mhfp.encoder import MHFPEncoder
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles, GetHashedMorganFingerprint
from rdkit.Chem import (
    rdMolDescriptors,
    MolFromSmiles,
    GetSymmSSSR,
    Descriptors,
    MACCSkeys,
)

from sklearn.preprocessing import StandardScaler

from molsetrep.encoders.encoder import Encoder


class SECMQNFPEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("SECMQNFPEncoder")

    def encode(
        self,
        smiles_sets: Iterable[Iterable[str]],
        labels_sets: Iterable[Iterable[Any]],
        label_dtype: Optional[torch.dtype] = None,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
        standardize: bool = True,
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        scaler = StandardScaler()

        fps_sets = []
        for smiles, labels in zip(smiles_sets, labels_sets):
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
                    # ds = []
                    # ds.append(Descriptors.MolLogP(submol))
                    # ds.append(Descriptors.TPSA(submol))
                    fp.append(np.array(ds))

                if standardize:
                    scaler.partial_fit(fp)

                fps.append(fp)
            fps_sets.append(fps)

        if standardize:
            for i in range(len(fps_sets)):
                for j in range(len(fps_sets[i])):
                    fps_sets[i][j] = scaler.transform(fps_sets[i][j])

        result = []

        for fps, labels in zip(fps_sets, labels_sets):
            result.append(super().to_tensor_dataset(fps, labels, label_dtype))

        return tuple(result)
