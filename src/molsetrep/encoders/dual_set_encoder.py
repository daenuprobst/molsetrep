from typing import Iterable, Any, Optional, Union

import torch
import numpy as np

from torch.utils.data import TensorDataset
from mhfp.encoder import MHFPEncoder
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles, GetHashedMorganFingerprint
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import (
    rdMolDescriptors,
    MolFromSmiles,
    GetSymmSSSR,
    Descriptors,
    MACCSkeys,
    BondType,
    HybridizationType,
    GetPeriodicTable,
)

PT = GetPeriodicTable()

from sklearn.preprocessing import StandardScaler

from molsetrep.encoders.encoder import Encoder


def one_hot_encode(prop: Any, vals: Union[int, Iterable[int]]):
    if not isinstance(vals, Iterable):
        vals = range(vals)

    result = [1 if prop == i else 0 for i in vals]

    if sum(result) == 0:
        result.append(1)
    else:
        result.append(0)

    return result


def get_atomic_invariants(atom):
    atomic_invariants = []
    atomic_invariants += one_hot_encode(atom.GetDegree(), 5)
    atomic_invariants += one_hot_encode(
        atom.GetExplicitValence()
        + atom.GetImplicitValence()
        - atom.GetNumExplicitHs()
        - atom.GetNumImplicitHs(),
        6,
    )
    atomic_invariants += one_hot_encode(atom.GetAtomicNum(), 100)
    atomic_invariants += one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    atomic_invariants += one_hot_encode(
        atom.GetChiralTag(),
        [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ],
    )
    atomic_invariants += one_hot_encode(atom.GetChiralTag(), 4)
    atomic_invariants.append(atom.GetMass())
    atomic_invariants.append(int(atom.IsInRing() == True))
    atomic_invariants.append(float(atom.GetProp("_GasteigerCharge")))
    # atomic_invariants.append(PT.GetRvdw(atom.GetAtomicNum()))

    total_hs = atom.GetNumExplicitHs() + atom.GetNumImplicitHs()
    atomic_invariants += one_hot_encode(total_hs, 6)

    return atomic_invariants


def get_bond_invariants(bond):
    bond_invariants = []

    atom_a = bond.GetBeginAtom()
    atom_b = bond.GetEndAtom()

    bond_invariants += one_hot_encode(
        bond.GetBondType(),
        [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ],
    )

    bond_invariants += one_hot_encode(int(bond.GetStereo()), 6)
    bond_invariants.append(int(bond.GetIsAromatic() == True))
    bond_invariants.append(int(bond.GetIsConjugated() == True))
    bond_invariants.append(bond.GetValenceContrib(atom_a))
    bond_invariants.append(bond.GetValenceContrib(atom_b))

    return bond_invariants


class DualSetEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("DualSetEncoder")

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_a = []
        fps_b = []
        fps_g = []
        for smi in smiles:
            mol = MolFromSmiles(smi)
            ComputeGasteigerCharges(mol)
            fp_atomic = []
            for atom in mol.GetAtoms():
                fp_atomic.append(get_atomic_invariants(atom))

            fp_bond = []
            for bond in mol.GetBonds():
                bond_invariants = get_bond_invariants(bond)

                atom_a = bond.GetBeginAtom()
                atom_b = bond.GetEndAtom()

                bond_invariants += one_hot_encode(atom_a.GetAtomicNum(), 100)
                bond_invariants += one_hot_encode(atom_b.GetAtomicNum(), 100)
                bond_invariants += one_hot_encode(atom_a.GetDegree(), 5)
                bond_invariants += one_hot_encode(atom_b.GetDegree(), 5)
                bond_invariants.append(int(atom_a.IsInRing() == True))
                bond_invariants.append(int(atom_b.IsInRing() == True))

                bond_invariants.append(float(atom_a.GetProp("_GasteigerCharge")))
                bond_invariants.append(float(atom_b.GetProp("_GasteigerCharge")))

                fp_bond.append(bond_invariants)

            if len(fp_bond) == 0:
                fp_bond.append([0] * (5 + 202 + 12 + 7 + 8))

            fps_a.append(fp_atomic)
            fps_b.append(fp_bond)

        return super().to_multi_tensor_dataset([fps_a, fps_b], labels, label_dtype)
