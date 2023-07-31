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


def get_mols(smi: Iterable[str]):
    parts = smi.split(">")

    reactants = [MolFromSmiles(s) for s in parts[0].split(".")]
    products = [MolFromSmiles(s) for s in parts[2].split(".")]

    if parts[1] != "":
        reactants += [MolFromSmiles(s) for s in parts[1].split(".")]

    return (reactants, products)


class RXNSetEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("RXNSetEncoder")

    def encode(
        self,
        rxn_smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_a_r = []
        fps_a_p = []
        # fps_b = []

        for smi in rxn_smiles:
            reactants, products = get_mols(smi)

            fp_atomic_r = []
            fp_atomic_p = []

            for mol in reactants:
                ComputeGasteigerCharges(mol)

                for atom in mol.GetAtoms():
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
                    atomic_invariants += one_hot_encode(
                        atom.GetFormalCharge(), [-2, -1, 0, 1, 2]
                    )
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

                    fp_atomic_r.append(atomic_invariants)

            for mol in products:
                ComputeGasteigerCharges(mol)

                for atom in mol.GetAtoms():
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
                    atomic_invariants += one_hot_encode(
                        atom.GetFormalCharge(), [-2, -1, 0, 1, 2]
                    )
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

                    fp_atomic_p.append(atomic_invariants)

            # fp_bond = []
            # for bond in mol.GetBonds():
            #     atom_a = bond.GetBeginAtom()
            #     atom_b = bond.GetEndAtom()

            #     bond_type = bond.GetBondType()
            #     bond_invariants = [
            #         bond_type == BondType.SINGLE,
            #         bond_type == BondType.DOUBLE,
            #         bond_type == BondType.TRIPLE,
            #         bond_type == BondType.AROMATIC,
            #     ]
            #     bond_invariants += one_hot_encode(int(bond.GetStereo()), 6)
            #     bond_invariants.append(int(bond.GetIsAromatic() == True))
            #     bond_invariants.append(int(bond.GetIsConjugated() == True))
            #     bond_invariants.append(bond.GetValenceContrib(atom_a))
            #     bond_invariants.append(bond.GetValenceContrib(atom_b))

            #     bond_invariants += one_hot_encode(atom_a.GetAtomicNum(), 100)
            #     bond_invariants += one_hot_encode(atom_b.GetAtomicNum(), 100)
            #     bond_invariants += one_hot_encode(atom_a.GetDegree(), 5)
            #     bond_invariants += one_hot_encode(atom_b.GetDegree(), 5)
            #     bond_invariants.append(int(atom_a.IsInRing() == True))
            #     bond_invariants.append(int(atom_b.IsInRing() == True))

            #     bond_invariants.append(float(atom_a.GetProp("_GasteigerCharge")))
            #     bond_invariants.append(float(atom_b.GetProp("_GasteigerCharge")))

            #     fp_bond.append(bond_invariants)

            # if len(fp_bond) == 0:
            #     fp_bond.append([0] * (4 + 202 + 12 + 7 + 8))

            fps_a_r.append(fp_atomic_r)
            fps_a_p.append(fp_atomic_p)
            # fps_b.append(fp_bond)

        return super().to_multi_tensor_dataset([fps_a_r, fps_a_p], labels, label_dtype)
