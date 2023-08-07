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
        mols: Iterable[Any],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_ligand = []
        fps_prot = []

        for ligand_mol, prot_mol in mols:
            # Handle ligand
            if self.charges:
                ComputeGasteigerCharges(ligand_mol)

            fp_ligand_atoms = []
            for atom in ligand_mol.GetAtoms():
                fp_ligand_atoms.append(get_atomic_invariants(atom, False))

            fps_ligand.append(fp_ligand_atoms)

            # handle protein
            if self.charges:
                ComputeGasteigerCharges(prot_mol)

            fp_prot_atoms = []
            for atom in prot_mol.GetAtoms():
                fp_prot_atoms.append(get_atomic_invariants(atom, False))

            fps_prot.append(fp_prot_atoms)

        return super().to_multi_tensor_dataset(
            [fps_ligand, fps_prot], labels, label_dtype
        )
