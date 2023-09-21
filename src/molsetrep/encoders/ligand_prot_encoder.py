from typing import Iterable, Any, Optional

import torch

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import MolFromSmiles

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import (
    get_atomic_invariants,
)


class LigandProtEncoder(Encoder):
    def __init__(self, coords: bool = True, charges: bool = False) -> Encoder:
        super().__init__("LigandProtEncoder")
        self.coords = coords
        self.charges = charges

    def encode(
        self,
        mols: Iterable[Any],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_ligand = []
        fps_prot = []

        for ligand_mol, prot_mol in mols:
            # Handle ligand
            if self.charges:
                ComputeGasteigerCharges(ligand_mol)

            conf = ligand_mol.GetConformer()

            fp_ligand_atoms = []
            for i, atom in enumerate(ligand_mol.GetAtoms()):
                xyz = []
                if self.coords:
                    pos = conf.GetAtomPosition(i)
                    xyz = [pos.x, pos.y, pos.z]

                fp_ligand_atoms.append(get_atomic_invariants(atom, False) + xyz)

            fps_ligand.append(fp_ligand_atoms)

            # handle protein
            if self.charges:
                ComputeGasteigerCharges(prot_mol)

            conf = prot_mol.GetConformer()

            fp_prot_atoms = []
            for i, atom in enumerate(prot_mol.GetAtoms()):
                xyz = []
                if self.coords:
                    pos = conf.GetAtomPosition(i)
                    xyz = [pos.x, pos.y, pos.z]

                fp_prot_atoms.append(get_atomic_invariants(atom, False) + xyz)

            fps_prot.append(fp_prot_atoms)

        return super().to_multi_tensor_dataset(
            [fps_ligand, fps_prot], labels, label_dtype
        )
