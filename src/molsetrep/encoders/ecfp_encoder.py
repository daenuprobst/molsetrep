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


class ECFPEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("ECFPEncoder")

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_a = []
        fps_b = []
        for smi in smiles:
            mol = MolFromSmiles(smi)
            fp_atomic = []
            for atom in mol.GetAtoms():
                atomic_invariants = [
                    atom.GetDegree(),
                    atom.GetExplicitValence()
                    + atom.GetImplicitValence()
                    - atom.GetNumExplicitHs()
                    - atom.GetNumImplicitHs(),
                    atom.GetAtomicNum(),
                    atom.GetMass(),
                    atom.GetFormalCharge(),
                    int(atom.IsInRing() == True),
                    atom.GetNumExplicitHs() + atom.GetNumImplicitHs(),
                ]
                fp_atomic.append(atomic_invariants)

            fp_bond = []
            for bond in mol.GetBonds():
                atom_a = bond.GetBeginAtom()
                atom_b = bond.GetEndAtom()

                bond_invariants = [
                    atom_a.GetAtomicNum(),
                    atom_b.GetAtomicNum(),
                    bond.GetBondTypeAsDouble(),
                    bond.GetIsAromatic(),
                ]

                fp_bond.append(bond_invariants)

            fps_a.append(fp_atomic)
            fps_b.append(fp_bond)

        return super().to_multi_tensor_dataset(fps_a, fps_b, labels, label_dtype)
