from typing import Iterable, Any, Optional

import torch

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import MolFromSmiles

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import (
    one_hot_encode,
    get_atomic_invariants,
    get_bond_invariants,
)


class DualSetEncoder(Encoder):
    def __init__(self, charges: bool = False) -> Encoder:
        super().__init__("DualSetEncoder")
        self.charges = charges

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_a = []
        fps_b = []
        for smi in smiles:
            mol = MolFromSmiles(smi)

            if self.charges:
                ComputeGasteigerCharges(mol)

            fp_atomic = []
            for atom in mol.GetAtoms():
                fp_atomic.append(get_atomic_invariants(atom, self.charges))

            fp_bond = []
            for bond in mol.GetBonds():
                bond_invariants = get_bond_invariants(bond)

                atom_a = bond.GetBeginAtom()
                atom_b = bond.GetEndAtom()

                bond_invariants += one_hot_encode(atom_a.GetAtomicNum(), 100)
                bond_invariants += one_hot_encode(atom_b.GetAtomicNum(), 100)
                bond_invariants += one_hot_encode(atom_a.GetTotalDegree(), 5)
                bond_invariants += one_hot_encode(atom_b.GetTotalDegree(), 5)
                bond_invariants.append(int(atom_a.IsInRing() == True))
                bond_invariants.append(int(atom_b.IsInRing() == True))

                if self.charges:
                    bond_invariants.append(float(atom_a.GetProp("_GasteigerCharge")))
                    bond_invariants.append(float(atom_b.GetProp("_GasteigerCharge")))

                fp_bond.append(bond_invariants)

            if len(fp_bond) == 0:
                l = 14 + 202 + 12 + 4

                if not self.charges:
                    l -= 2

                fp_bond.append([0] * l)

            fps_a.append(fp_atomic)
            fps_b.append(fp_bond)

        return super().to_multi_tensor_dataset([fps_a, fps_b], labels, label_dtype)
