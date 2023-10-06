from typing import Iterable, Any, Optional, Tuple

import torch
import numpy as np

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import MolFromSmiles

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import get_atomic_invariants, one_hot_encode


class LigandProtPairEncoder(Encoder):
    def __init__(self, charges: bool = False) -> Encoder:
        super().__init__("LigandProtPairEncoder")
        self.charges = charges

    def get_neighbours(
        self, query: np.ndarray, x: np.ndarray, r: float = 2.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        dists = np.linalg.norm(x - query, axis=1)
        inds = np.where(dists < r)[0]
        return inds, dists[inds]

    def encode(
        self,
        mols: Iterable[Any],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_atom_pairs = []

        for ligand_mol, prot_mol in mols:
            # Handle ligand
            if self.charges:
                ComputeGasteigerCharges(prot_mol)

            conf_ligand = ligand_mol.GetConformer()
            conf_prot = prot_mol.GetConformer()

            pos_ligand = conf_ligand.GetPositions()
            pos_prot = conf_prot.GetPositions()

            fp_atom_pair = []

            # If r is to small (no atoms added for the whole complex) increase r
            increase_r = 0.0
            while len(fp_atom_pair) == 0:
                for i, ligand_xyz in enumerate(pos_ligand):
                    neighbour_inds, neighbour_dists = self.get_neighbours(
                        ligand_xyz, pos_prot, 5.5 + increase_r
                    )
                    if len(neighbour_inds) == 0:
                        continue

                    atom_ligand = ligand_mol.GetAtomWithIdx(i)

                    for idx, dist in zip(neighbour_inds, neighbour_dists):
                        atom_prot = prot_mol.GetAtomWithIdx(int(idx))

                        fp_atom_pair.append(
                            get_atomic_invariants(atom_ligand, self.charges)
                            + one_hot_encode(int(round(dist)), 6)
                            + get_atomic_invariants(atom_prot, self.charges)
                        )

                increase_r += 0.25

            fps_atom_pairs.append(fp_atom_pair)

        return super().to_multi_tensor_dataset([fps_atom_pairs], labels, label_dtype)
