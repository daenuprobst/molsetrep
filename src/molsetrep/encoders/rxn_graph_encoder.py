import random
from typing import Optional, Any, Iterable
from collections import defaultdict
from multiprocessing import cpu_count

import torch
import torch_geometric
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit import RDLogger
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from molsetrep.encoders.common import (
    get_atomic_invariants_as_dict,
    get_bond_invariants_as_dict,
    N_BOND_INVARIANTS,
)


class RXNGraphEncoder:
    def __init__(self, charges: bool = False, fix_seed=True) -> "RXNGraphEncoder":
        self.charges = charges
        self.fix_seed = fix_seed

    def get_mols(self, smi: str):
        parts = smi.split(">")

        reactants = [MolFromSmiles(s) for s in parts[0].split(".")]
        products = [MolFromSmiles(s) for s in parts[2].split(".")]

        if parts[1] != "":
            reactants += [MolFromSmiles(s) for s in parts[1].split(".")]

        return reactants, products

    def mol_to_nx(self, mol: Mol) -> nx.Graph:
        G = nx.Graph()

        if self.charges:
            ComputeGasteigerCharges(mol)

        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(), **get_atomic_invariants_as_dict(atom, self.charges)
            )

        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                **get_bond_invariants_as_dict(bond),
            )

        return G

    def mols_to_nx(self, mols: Iterable[Mol], is_product) -> nx.Graph:
        G = nx.Graph()

        node_idx = 0
        atom_to_node_map = {}
        for mol_idx, mol in enumerate(mols):
            if self.charges:
                ComputeGasteigerCharges(mol)

            for atom in mol.GetAtoms():
                atom_to_node_map[atom.GetIdx()] = node_idx
                G.add_node(
                    node_idx,
                    **get_atomic_invariants_as_dict(atom, self.charges),
                    is_product=is_product[mol_idx],
                )
                node_idx += 1

            for bond in mol.GetBonds():
                G.add_edge(
                    atom_to_node_map[bond.GetBeginAtomIdx()],
                    atom_to_node_map[bond.GetEndAtomIdx()],
                    **get_bond_invariants_as_dict(bond),
                )

        return G

    def smiles_to_nx(self, smiles: str) -> nx.Graph:
        mol = MolFromSmiles(smiles)
        return self.mol_to_nx(mol)

    def rxn_smiles_to_nx(self, rxn_smiles: str) -> nx.Graph:
        reactants, products = self.get_mols(rxn_smiles)
        is_product = {
            i: 1 if i < len(reactants) else 0
            for i in range(len(reactants) + len(products))
        }

        return self.mols_to_nx(reactants + products, is_product)

    def nx_to_pyg(
        self, G: Any, y: Optional[Any] = None, idx: Optional[int] = None
    ) -> torch_geometric.data.Data:
        r"""Adapted from torch.utils.from_networkx.
        Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            group_node_attrs (List[str] or all, optional): The node attributes to
                be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
            group_edge_attrs (List[str] or all, optional): The edge attributes to
                be concatenated and added to :obj:`data.edge_attr`.
                (default: :obj:`None`)

        .. note::

            All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
            be numeric.

        Examples:

            >>> edge_index = torch.tensor([
            ...     [0, 1, 1, 2, 2, 3],
            ...     [1, 0, 2, 1, 3, 2],
            ... ])
            >>> data = Data(edge_index=edge_index, num_nodes=4)
            >>> g = to_networkx(data)
            >>> # A `Data` object is returned
            >>> from_networkx(g)
            Data(edge_index=[2, 6], num_nodes=4)
        """

        G = G.to_directed() if not nx.is_directed(G) else G

        mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
        edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
        for i, (src, dst) in enumerate(G.edges()):
            edge_index[0, i] = mapping[src]
            edge_index[1, i] = mapping[dst]

        data = defaultdict(list)

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError("Not all nodes contain the same attributes")
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError("Not all edges contain the same attributes")
            for key, value in feat_dict.items():
                key = f"edge_{key}" if key in node_attrs else key
                data[str(key)].append(value)

        for key, value in G.graph.items():
            if key == "node_default" or key == "edge_default":
                continue  # Do not load default attributes.
            key = f"graph_{key}" if key in node_attrs else key
            data[str(key)] = value

        for key, value in data.items():
            if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
                data[key] = torch.stack(value, dim=0)
            else:
                try:
                    data[key] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass

        data["edge_index"] = edge_index.view(2, -1)

        if data["edge_index"].shape[1] == 0:
            return None
            # edge_attrs = [f"edge_{i}" for i in range(N_BOND_INVARIANTS)]
            # for i in range(N_BOND_INVARIANTS):
            #     data[f"edge_{i}"] = torch.tensor([])

        data = Data.from_dict(data)

        xs = []
        for key in list(node_attrs):
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]

        data.x = torch.cat(xs, dim=-1)

        xs = []
        for key in list(edge_attrs):
            key = f"edge_{key}" if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]

        data.edge_attr = torch.cat(xs, dim=-1)

        if data.x is None and data.pos is None:
            data.num_nodes = G.number_of_nodes()

        if y is not None:
            data.y = y

        if idx is not None:
            data.idx = idx

        return data

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        batch_size: int = 64,
        atom_attrs: Optional[Iterable[str]] = None,
        bond_attrs: Optional[Iterable[str]] = None,
        suppress_rdkit_warnings: bool = True,
        label_dtype: torch.dtype = None,
        shuffle: bool = False,
        **kwargs,
    ):
        if suppress_rdkit_warnings:
            RDLogger.DisableLog("rdApp.*")

        if atom_attrs is None:
            atom_attrs = []

        if bond_attrs is None:
            bond_attrs = []

        data_list = []
        for i, G in tqdm(
            enumerate([self.rxn_smiles_to_nx(s) for s in smiles]),
            total=len(smiles),
        ):
            data = self.nx_to_pyg(G, y=torch.tensor([labels[i]], dtype=label_dtype))
            if data is not None:
                data_list.append(data)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        data_loader = DataLoader(
            data_list,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            worker_init_fn=seed_worker if self.fix_seed else None,
            generator=g if self.fix_seed else None,
            num_workers=cpu_count() if cpu_count() < 8 else 8,
        )

        return data_loader
