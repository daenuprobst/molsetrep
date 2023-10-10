from typing import Iterable, Any, Optional

import torch

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import MolFromSmiles

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import get_atomic_invariants


class SingleSetEncoder(Encoder):
    def __init__(self, charges: bool = False) -> Encoder:
        super().__init__("SingleSetEncoder")
        self.charges = charges

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_a = []
        for smi in smiles:
            mol = MolFromSmiles(smi)

            if self.charges:
                ComputeGasteigerCharges(mol)

            fp_atomic = []
            for atom in mol.GetAtoms():
                fp_atomic.append(get_atomic_invariants(atom, self.charges))

            fps_a.append(fp_atomic)

        return super().to_multi_tensor_dataset([fps_a], labels, label_dtype)
