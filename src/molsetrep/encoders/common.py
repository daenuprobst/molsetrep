from typing import Iterable, Any, Union

from rdkit.Chem import (
    BondType,
    HybridizationType,
)

N_BOND_INVARIANTS = 14

RESIDUE_MAP = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLU": 6,
    "GLN": 7,
    "GLY": 8,
    "HIS": 9,
    "HYP": 10,
    "ILE": 11,
    "LEU": 12,
    "LYS": 13,
    "MET": 14,
    "PHE": 15,
    "PRO": 16,
    "GLP": 17,
    "SER": 18,
    "THR": 19,
    "TRP": 20,
    "TYR": 21,
    "VAL": 22,
}


def one_hot_encode(prop: Any, vals: Union[int, Iterable[int]]):
    if not isinstance(vals, Iterable):
        vals = range(vals)

    result = [1 if prop == i else 0 for i in vals]

    if sum(result) == 0:
        result.append(1)
    else:
        result.append(0)

    return result


def get_atomic_invariants_as_dict(atom, charges: bool = True):
    invariants = get_atomic_invariants(atom, charges)
    return {f"node_{i}": v for i, v in enumerate(invariants)}


def get_bond_invariants_as_dict(bond):
    invariants = get_bond_invariants(bond)
    return {f"edge_{i}": v for i, v in enumerate(invariants)}


def get_atomic_invariants(atom, charges: bool = True, max_atomic_num: int = 100):
    atomic_invariants = []
    atomic_invariants += one_hot_encode(atom.GetTotalDegree(), 6)
    atomic_invariants += one_hot_encode(atom.GetAtomicNum(), max_atomic_num)
    atomic_invariants += one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    atomic_invariants += one_hot_encode(
        atom.GetHybridization(),
        [
            HybridizationType.S,
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP2D,
            HybridizationType.SP3,
            HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ],
    )

    atomic_invariants += one_hot_encode(atom.GetChiralTag(), 4)
    atomic_invariants.append(int(atom.IsInRing() == True))

    if charges:
        atomic_invariants.append(float(atom.GetProp("_GasteigerCharge")))

    atomic_invariants += one_hot_encode(atom.GetTotalNumHs(), 6)
    atomic_invariants.append(0.01 * atom.GetMass())

    return atomic_invariants


def get_bond_invariants(bond):
    bond_invariants = []

    # atom_a = bond.GetBeginAtom()
    # atom_b = bond.GetEndAtom()

    bond_invariants += one_hot_encode(
        bond.GetBondType(),
        [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ],
    )

    bond_invariants += one_hot_encode(int(bond.GetStereo()), 6)
    bond_invariants.append(int(bond.GetIsAromatic() == True))
    bond_invariants.append(int(bond.GetIsConjugated() == True))
    # bond_invariants += one_hot_encode(int(bond.GetValenceContrib(atom_a)), 3)
    # bond_invariants += one_hot_encode(int(bond.GetValenceContrib(atom_b)), 3)

    return bond_invariants


def get_ring_invariants(ring, mol):
    ring_invariants = []

    atom = mol.GetAtomWithIdx(ring[0])
    ring_invariants += one_hot_encode(len(ring), 9)
    ring_invariants.append(int(atom.GetIsAromatic()))

    return ring_invariants
