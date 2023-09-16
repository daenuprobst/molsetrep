from typing import Iterable, Any, Optional, Union

import torch

from torch.utils.data import TensorDataset
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem import MolFromSmiles

from rdkit.Chem.AllChem import GetMorganGenerator

from tqdm import tqdm
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
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fpgen = GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True)

        fps_r = []
        fps_p = []

        for smi in tqdm(rxn_smiles):
            reactants, products = get_mols(smi)

            fp_ecfp_r = []
            fp_ecfp_p = []

            for mol in reactants:
                if mol is not None:
                    fp_ecfp_r.append(fpgen.GetFingerprint(mol))

            for mol in products:
                if mol is not None:
                    fp_ecfp_p.append(fpgen.GetFingerprint(mol))

            fps_r.append(fp_ecfp_r)
            fps_p.append(fp_ecfp_p)

        return super().to_multi_tensor_dataset([fps_r, fps_p], labels, label_dtype)
