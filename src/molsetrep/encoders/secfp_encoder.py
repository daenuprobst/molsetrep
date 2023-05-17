from typing import List
from mhfp.encoder import MHFPEncoder
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdchem import Mol

from molsetrep.encoders.encoder import Encoder


class SECFPEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("SECFPEncoder")

    def encode(
        mol: Mol,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
    ) -> List[Mol]:
        substructs = set(
            MHFPEncoder.shingling_from_mol(mol, radius, rings, kekulize, min_radius)
        )

        return [MolFromSmiles(s) for s in substructs]
