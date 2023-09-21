from typing import Iterable, Any, Optional

import torch
import numpy as np

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import MolFromSmiles

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import (
    get_atomic_invariants,
)


class LigandProtSpatialEncoder(Encoder):
    def __init__(self, charges: bool = False) -> Encoder:
        super().__init__("LigandProtEncoder")
        self.charges = charges

    def get_neighbours(
        self, query: np.ndarray, x: np.ndarray, r: float = 2.5
    ) -> np.ndarray:
        dists = np.linalg.norm(x - query, axis=1)
        return np.where(dists < r)[0]

    def encode(
        self,
        mols: Iterable[Any],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_atoms = []

        for ligand_mol, prot_mol in mols:
            # Handle ligand
            if self.charges:
                ComputeGasteigerCharges(prot_mol)

            conf_ligand = ligand_mol.GetConformer()
            conf_prot = prot_mol.GetConformer()

            pos_ligand = conf_ligand.GetPositions()
            pos_prot = conf_prot.GetPositions()

            fp_atoms = []

            # If r is to small (no atoms added for the whole complex) increase r
            increase_r = 0.0
            while len(fp_atoms) == 0:
                for i, ligand_xyz in enumerate(pos_ligand):
                    neighbour_inds = self.get_neighbours(
                        ligand_xyz, pos_prot, 3.5 + increase_r
                    )
                    if len(neighbour_inds) == 0:
                        continue

                    # print(len(neighbour_inds))

                    fp_atoms.append(
                        get_atomic_invariants(
                            ligand_mol.GetAtomWithIdx(i), self.charges
                        )
                    )

                    for idx in neighbour_inds:
                        fp_atoms.append(
                            get_atomic_invariants(
                                prot_mol.GetAtomWithIdx(int(idx)), self.charges
                            )
                        )
                increase_r += 0.25

            fps_atoms.append(fp_atoms)

            # fp_ligand_atoms = []
            # for i, atom in enumerate(ligand_mol.GetAtoms()):
            #     xyz = []
            #     if self.coords:
            #         pos = conf.GetAtomPosition(i)
            #         xyz = [pos.x, pos.y, pos.z]

            #     fp_ligand_atoms.append(get_atomic_invariants(atom, False) + xyz)

            # fps_ligand.append(fp_ligand_atoms)

            # # handle protein
            # if self.charges:
            #     ComputeGasteigerCharges(prot_mol)

            # conf = prot_mol.GetConformer()

            # fp_prot_atoms = []
            # for i, atom in enumerate(prot_mol.GetAtoms()):
            #     xyz = []
            #     if self.coords:
            #         pos = conf.GetAtomPosition(i)
            #         xyz = [pos.x, pos.y, pos.z]

            #     fp_prot_atoms.append(get_atomic_invariants(atom, False) + xyz)

            # fps_prot.append(fp_prot_atoms)

        return super().to_multi_tensor_dataset([fps_atoms], labels, label_dtype)
