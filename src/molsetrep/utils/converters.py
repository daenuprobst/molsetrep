from collections import defaultdict
import networkx as nx


def __get_atom_props(atom):
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


def __get_bond_props(bond):
    return {
        "bond_type": bond.GetBondTypeAsDouble(),
        "bond_conjugated": int(bond.GetIsConjugated()),
        "bond_stereo": int(bond.GetStereo()),
    }


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), **__get_atom_props(atom))

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **__get_bond_props(bond)
        )

    return G


def mols_to_nx(mols):
    G = nx.Graph()

    atom_idx = 0
    graph_to_mol = defaultdict(list)
    node_idx_to_mol = {}
    for mol_idx, mol in enumerate(mols):
        atom_idx_map = {}
        for atom in mol.GetAtoms():
            G.add_node(atom_idx, **__get_atom_props(atom))
            atom_idx_map[atom.GetIdx()] = atom_idx
            node_idx_to_mol[atom_idx] = mol_idx
            atom_idx += 1
            graph_to_mol[mol_idx].append(atom_idx)

        for bond in mol.GetBonds():
            G.add_edge(
                atom_idx_map[bond.GetBeginAtomIdx()],
                atom_idx_map[bond.GetEndAtomIdx()],
                **__get_bond_props(bond)
            )

    return G, graph_to_mol, node_idx_to_mol
