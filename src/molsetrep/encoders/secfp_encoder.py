from typing import Iterable, Any, Optional

import torch
import numpy as np

from tqdm import tqdm

from torch.utils.data import TensorDataset
from mhfp.encoder import MHFPEncoder
from rdkit.Chem.AllChem import MolFromSmiles, GetSymmSSSR

from karateclub.graph_embedding import GL2Vec, FeatherGraph
from karateclub.estimator import Estimator

from molsetrep.encoders.encoder import Encoder
from molsetrep.utils.converters import mol_to_nx


class SECFPEncoder(Encoder):
    def __init__(self) -> Encoder:
        super().__init__("SECFPEncoder")

    @staticmethod
    def pretrain(
        smiles: Iterable[str],
        graph_embedder: Estimator,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
    ) -> Estimator:
        return SECFPEncoder.fit(
            smiles, graph_embedder, radius, rings, kekulize, min_radius
        )

    @staticmethod
    def fit(
        smiles: Iterable[str],
        graph_embedder: Estimator = None,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
    ) -> Estimator:
        if graph_embedder is None:
            graph_embedder = FeatherGraph()

        substructs = set()

        for smi in tqdm(
            smiles,
            "Fitting (extracting substructures)",
            total=len(smiles),
            leave=False,
        ):
            substructs.update(
                MHFPEncoder.shingling_from_mol(
                    MolFromSmiles(smi), radius, rings, kekulize, min_radius
                )
            )

        graphs = []
        for s in tqdm(
            substructs,
            "Fitting (converting to graphs)",
            total=len(substructs),
            leave=False,
        ):
            mol = MolFromSmiles(s, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            GetSymmSSSR(mol)
            graphs.append(mol_to_nx(mol))

        graph_embedder.fit(graphs)
        return graph_embedder

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
        graph_embedder: Optional[Estimator] = None,
        pretrained_graph_embedder: bool = False,
    ) -> TensorDataset:
        if graph_embedder is None:
            graph_embedder = GL2Vec()

        if not pretrained_graph_embedder:
            graph_embedder = SECFPEncoder.fit(
                smiles, graph_embedder, radius, rings, kekulize, min_radius
            )

        fps = []

        for smi in tqdm(smiles, "Embedding", total=len(smiles), leave=False):
            substructs = set(
                MHFPEncoder.shingling_from_mol(
                    MolFromSmiles(smi), radius, rings, kekulize, min_radius
                )
            )

            graphs = []
            for s in substructs:
                mol = MolFromSmiles(s, sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                GetSymmSSSR(mol)
                graphs.append(mol_to_nx(mol))

            fp = graph_embedder.infer(graphs)
            fps.append(fp)

        return super().to_tensor_dataset(fps, labels, label_dtype)
