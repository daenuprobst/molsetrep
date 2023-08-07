from deepchem.feat.base_classes import ComplexFeaturizer
from molsetrep.encoders.common import get_atomic_invariants
from deepchem.utils.rdkit_utils import load_molecule


class PDBBindFeaturizer(ComplexFeaturizer):
    def __init__(self):
        ...

    def _featurize(self, datapoint, **kwargs):
        mol_pdb_file, protein_pdb_file = datapoint

        mol_coords, ob_mol = load_molecule(
            mol_pdb_file, calc_charges=False, add_hydrogens=False
        )

        protein_coords, protein_mol = load_molecule(
            protein_pdb_file, calc_charges=False, add_hydrogens=False
        )

        return (ob_mol, protein_mol)
