from typing import Iterable, Any, Optional

import torch

from torch.utils.data import TensorDataset
from mhfp.encoder import MHFPEncoder
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import (
    rdMolDescriptors,
    MolFromSmiles,
    GetSSSR,
    Descriptors,
)

from descriptastorus.descriptors import rdNormalizedDescriptors

from molsetrep.encoders.encoder import Encoder
from molsetrep.encoders.common import (
    one_hot_encode,
    get_atomic_invariants,
    get_bond_invariants,
    get_ring_invariants,
)


class TripleSetEncoder(Encoder):
    def __init__(self, charges: bool = True) -> Encoder:
        super().__init__("TripleSetEncoder")
        self.charges = charges

    def encode(
        self,
        smiles: Iterable[str],
        labels: Iterable[Any],
        label_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> TensorDataset:
        RDLogger.DisableLog("rdApp.*")

        fps_a = []
        fps_b = []
        fps_g = []
        for smi in smiles:
            mol = MolFromSmiles(smi)

            if self.charges:
                ComputeGasteigerCharges(mol)

            fp_atomic = []
            for atom in mol.GetAtoms():
                fp_atomic.append(get_atomic_invariants(atom, self.charges))

            fp_bond = []
            for bond in mol.GetBonds():
                bond_invariants = get_bond_invariants(bond)

                atom_a = bond.GetBeginAtom()
                atom_b = bond.GetEndAtom()

                bond_invariants += one_hot_encode(atom_a.GetAtomicNum(), 100)
                bond_invariants += one_hot_encode(atom_b.GetAtomicNum(), 100)
                bond_invariants += one_hot_encode(atom_a.GetDegree(), 5)
                bond_invariants += one_hot_encode(atom_b.GetDegree(), 5)
                bond_invariants.append(int(atom_a.IsInRing() == True))
                bond_invariants.append(int(atom_b.IsInRing() == True))

                if self.charges:
                    bond_invariants.append(float(atom_a.GetProp("_GasteigerCharge")))
                    bond_invariants.append(float(atom_b.GetProp("_GasteigerCharge")))

                fp_bond.append(bond_invariants)

            if len(fp_bond) == 0:
                l = 5 + 202 + 12 + 7 + 8

                if not self.charges:
                    l -= 2

                fp_bond.append([0] * l)

            # norm_2d_gen = rdNormalizedDescriptors.RDKit2DNormalized()
            # feats = norm_2d_gen.process(smi)[1:]
            fp_global = []

            for ring in GetSSSR(mol):
                fp_global.append(get_ring_invariants(list(ring), mol))

            if len(fp_global) == 0:
                fp_global.append([0] * 11)

            # for sh in MHFPEncoder.shingling_from_mol(mol, min_radius=0):
            #     sh_mol = MolFromSmiles(sh, sanitize=True)

            #     if sh_mol:
            #         norm_2d_gen = rdNormalizedDescriptors.RDKit2DNormalized()
            #         feats = norm_2d_gen.process(sh)[1:]
            #         fp_global.append(
            #             feats
            #             # [
            #             #     Descriptors.MolLogP(sh_mol),
            #             #     rdMolDescriptors.CalcExactMolWt(sh_mol),
            #             #     rdMolDescriptors.CalcTPSA(sh_mol),
            #             #     rdMolDescriptors.CalcExactMolWt(sh_mol),
            #             #     rdMolDescriptors.CalcTPSA(sh_mol),
            #             #     rdMolDescriptors.CalcPhi(sh_mol),
            #             #     rdMolDescriptors.CalcKappa1(sh_mol),
            #             #     rdMolDescriptors.CalcKappa2(sh_mol),
            #             #     rdMolDescriptors.CalcKappa3(sh_mol),
            #             #     rdMolDescriptors.CalcChi0n(sh_mol),
            #             #     rdMolDescriptors.CalcChi0v(sh_mol),
            #             #     rdMolDescriptors.CalcChi1n(sh_mol),
            #             #     rdMolDescriptors.CalcChi1v(sh_mol),
            #             #     rdMolDescriptors.CalcChi2n(sh_mol),
            #             #     rdMolDescriptors.CalcChi2v(sh_mol),
            #             # ]
            #             # + one_hot_encode(rdMolDescriptors.CalcNumRings(sh_mol), 9)
            #             # + one_hot_encode(
            #             #     rdMolDescriptors.CalcNumAromaticRings(sh_mol), 9
            #             # )
            #             # + rdMolDescriptors.CalcAUTOCORR2D(sh_mol)
            #             # + list(rdMolDescriptors.CalcCrippenDescriptors(sh_mol)),
            #         )
            #     else:
            #         fp_global.append([0] * 200)

            fps_a.append(fp_atomic)
            fps_b.append(fp_bond)
            fps_g.append(fp_global)

        return super().to_multi_tensor_dataset(
            [fps_a, fps_b, fps_g], labels, label_dtype
        )
