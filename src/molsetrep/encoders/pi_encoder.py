from typing import Iterable, Any, Optional

import torch

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import ResonanceMolSupplier

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import (
    one_hot_encode,
    get_atomic_invariants,
    get_bond_invariants,
)


class PiEncoder(Encoder):
    def __init__(self, charges: bool = True) -> Encoder:
        super().__init__("PiEncoder")
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
        fps_api = []
        for smi in smiles:
            mol = MolFromSmiles(smi)

            if self.charges:
                ComputeGasteigerCharges(mol)

            fp_atomic = []
            fp_atomic_pi = []
            rsuppl = ResonanceMolSupplier(mol)
            for idx, atom in enumerate(mol.GetAtoms()):
                conj_group = rsuppl.GetAtomConjGrpIdx(idx)
                if conj_group >= 0:
                    fp_atomic_pi.append(get_atomic_invariants(atom, self.charges))
                else:
                    fp_atomic.append(get_atomic_invariants(atom, self.charges))

            if len(fp_atomic) == 0:
                l = 133

                if self.charges:
                    l += 2

                fp_atomic.append([0] * l)

            if len(fp_atomic_pi) == 0:
                l = 133

                if self.charges:
                    l += 2

                fp_atomic_pi.append([0] * l)

            fps_a.append(fp_atomic)
            fps_api.append(fp_atomic_pi)

        return super().to_multi_tensor_dataset([fps_a, fps_api], labels, label_dtype)
