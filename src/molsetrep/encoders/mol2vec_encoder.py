from typing import Iterable, Any, Optional, List

import torch
import numpy as np

from gensim.models import word2vec
from torch.utils.data import TensorDataset
from mhfp.encoder import MHFPEncoder
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles, GetMorganFingerprint

from molsetrep.internals import DataManager
from molsetrep.encoders.encoder import Encoder


class Mol2VecEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("Mol2VecEncoder")

        dm = DataManager()
        if not dm.exists("mol2vec_model_300dim.pt"):
            dm.download_file(
                "https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz"
            )

        self.model = word2vec.Word2Vec.load(
            str(dm.get_path("mol2vec_model_300dim.pkl"))
        )

    def mol2alt_sentence(self, smiles):
        # Copied from https://github.com/samoturk/mol2vec/blob/850d944d5f48a58e26ed0264332b5741f72555aa/mol2vec/features.py#L129-L168
        mol = MolFromSmiles(smiles)
        radii = list(range(2))
        info = {}
        _ = GetMorganFingerprint(
            mol, 1, bitInfo=info
        )  # info: dictionary identifier, atom_idx, radius

        mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
        dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

        for element in info:
            for atom_idx, radius_at in info[element]:
                dict_atoms[atom_idx][
                    radius_at
                ] = element  # {atom number: {fp radius: identifier}}

        # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
        identifiers_alt = []
        for atom in dict_atoms:  # iterate over atoms
            for r in radii:  # iterate over radii
                identifiers_alt.append(dict_atoms[atom][r])

        alternating_sentence = map(str, [x for x in identifiers_alt if x])

        return list(alternating_sentence)

    def sentences2vec(self, sentences: List, unseen="UNK") -> np.ndarray:
        keys = set(self.model.wv.key_to_index.keys())

        vec = []
        if unseen:
            unseen_vec = self.model.wv.get_vector(unseen)

        for sentence in sentences:
            if unseen:
                vec.append(
                    sum(
                        [
                            self.model.wv.get_vector(y)
                            if y in set(sentence) & keys
                            else unseen_vec
                            for y in sentence
                        ]
                    )
                )
            else:
                vec.append(
                    sum(
                        [
                            self.model.wv.get_vector(y)
                            for y in sentence
                            if y in set(sentence) & keys
                        ]
                    )
                )
        return np.array(vec)

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps = []
        for smi in smiles:
            sentence = self.mol2alt_sentence(smi)
            sentence_set = self.sentences2vec([sentence])[0]
            fps.append(sentence_set)

        return super().to_tensor_dataset(fps, labels, label_dtype)
