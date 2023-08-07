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


class LigandProtEncoder(Encoder):
    def __init__(self, charges: bool = True) -> Encoder:
        super().__init__("LigandProtEncoder")
        self.charges = charges

    def encode(
        self,
        ligand_mols: Iterable[Any],
        prot_mols: Iterable[Any],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_ligand = []
        fps_prot = []
        for mol in ligand_mols:
            if self.charges:
                ComputeGasteigerCharges(mol)

            fp_ligand_atoms = []
            for atom in mol.GetAtoms():
                fp_ligand_atoms.append(get_atomic_invariants(atom, False))

            fps_ligand.append(fp_ligand_atoms)

        for mol in prot_mols:
            if self.charges:
                ComputeGasteigerCharges(mol)

            fp_prot_atoms = []
            for atom in mol.GetAtoms():
                fp_prot_atoms.append(get_atomic_invariants(atom, False))

            fps_prot.append(fp_prot_atoms)

        return super().to_multi_tensor_dataset(
            [fps_ligand, fps_prot], labels, label_dtype
        )
