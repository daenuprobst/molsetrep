from typing import Optional, Union, List, Any, Iterable, Dict
from collections import defaultdict

import torch
import torch_geometric
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, ImbalancedSampler
from rdkit.Chem.rdchem import Mol, Atom, Bond
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit import RDLogger


def __get_atom_props(atom: Atom) -> Dict[str, Any]:
    return {
        "atomic_num": atom.GetAtomicNum(),
        "charge": atom.GetFormalCharge(),
        "aromatic": int(atom.GetIsAromatic() == True),
        "is_in_ring": int(atom.IsInRing() == True),
        "hydrogen_count": atom.GetTotalNumHs(),
        "hybridization": int(atom.GetHybridization()),
        "chiral_tag": int(atom.GetChiralTag()),
        "degree": atom.GetDegree(),
        "radical_count": atom.GetNumRadicalElectrons(),
    }


def __get_bond_props(bond: Bond) -> Dict[str, Any]:
    return {
        "bond_type": bond.GetBondTypeAsDouble(),
        "bond_conjugated": int(bond.GetIsConjugated()),
        "bond_stereo": int(bond.GetStereo()),
    }


def mol_to_nx(mol: Mol) -> nx.Graph:
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), **__get_atom_props(atom))

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **__get_bond_props(bond)
        )

    return G


def smiles_to_nx(smiles: str) -> nx.Graph:
    mol = MolFromSmiles(smiles)
    return mol_to_nx(mol)


def nx_to_pyg(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
    y: Optional[Any] = None,
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

    if len(data["atomic_num"]) == 1:
        data["edge_index"] = torch.tensor([[0], [0]])
        data["bond_type"] = torch.tensor([0])
        data["bond_conjugated"] = torch.tensor([0])
        data["bond_stereo"] = torch.tensor([0])

    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
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

    return data


def molnet_to_pyg(
    train,
    valid,
    test,
    task: int = 0,
    batch_size: int = 64,
    atom_attrs: Optional[Iterable[str]] = None,
    bond_attrs: Optional[Iterable[str]] = None,
    suppress_rdkit_warnings: bool = True,
    label_type: torch.dtype = None,
    imbalanced_sampler: bool = False,
):
    if suppress_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    if atom_attrs is None:
        atom_attrs = [
            "atomic_num",
            "charge",
            "aromatic",
            "is_in_ring",
            "hydrogen_count",
            "hybridization",
            "chiral_tag",
            "degree",
            "radical_count",
        ]

    if bond_attrs is None:
        bond_attrs = [
            "bond_type",
            "bond_conjugated",
            "bond_stereo",
        ]

    train_data_list = []
    for i, G in enumerate([smiles_to_nx(s) for s in train.ids]):
        train_data_list.append(
            nx_to_pyg(
                G,
                group_node_attrs=atom_attrs,
                group_edge_attrs=bond_attrs,
                y=torch.tensor([train.y[i][task]], dtype=label_type),
            )
        )

    valid_data_list = []
    for i, G in enumerate([smiles_to_nx(s) for s in valid.ids]):
        valid_data_list.append(
            nx_to_pyg(
                G,
                group_node_attrs=atom_attrs,
                group_edge_attrs=bond_attrs,
                y=torch.tensor([valid.y[i][task]], dtype=label_type),
            )
        )

    test_data_list = []
    for i, G in enumerate([smiles_to_nx(s) for s in test.ids]):
        test_data_list.append(
            nx_to_pyg(
                G,
                group_node_attrs=atom_attrs,
                group_edge_attrs=bond_attrs,
                y=torch.tensor([test.y[i][task]], dtype=label_type),
            )
        )

    sampler = None
    shuffle = True
    if imbalanced_sampler:
        sampler = ImbalancedSampler(train_data_list)
        shuffle = False

    train_loader = DataLoader(
        train_data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        sampler=sampler,
    )
    valid_loader = DataLoader(
        valid_data_list, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_data_list, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader, valid_loader, test_loader
