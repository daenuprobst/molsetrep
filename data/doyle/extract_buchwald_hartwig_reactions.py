# %%
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from drfp import DrfpEncoder


# %%
def generate_buchwald_hartwig_rxns(df):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = "[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]"
    methylaniline = "Cc1ccc(N)cc1"
    pd_catalyst = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []

    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row["Aryl halide"]), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])

    df["product"] = products
    rxns = []

    for i, row in df.iterrows():
        reactants = Chem.MolToSmiles(
            Chem.MolFromSmiles(
                f"{row['Aryl halide']}.{methylaniline}.{pd_catalyst}.{row['Ligand']}.{row['Base']}.{row['Additive']}"
            )
        )
        rxns.append(f"{reactants.replace('N~', '[NH2]')}>>{row['product']}")

    return rxns


# %%
NAMES = [
    "FullCV_01",
    "FullCV_02",
    "FullCV_03",
    "FullCV_04",
    "FullCV_05",
    "FullCV_06",
    "FullCV_07",
    "FullCV_08",
    "FullCV_09",
    "FullCV_10",
    "Test1",
    "Test2",
    "Test3",
    "Test4",
]


# %%
y_predictions = []
y_tests = []
r2_scores = []
rmse_scores = []

for name in NAMES:
    df_doyle = pd.read_excel(
        "../../data/Dreher_and_Doyle_input_data.xlsx",
        sheet_name=name,
        engine="openpyxl",
    )

    df_doyle["rxn"] = generate_buchwald_hartwig_rxns(df_doyle)

    data = df_doyle[["rxn", "Output"]]  # paper has starting index 1 not 0

    data.columns = ["smiles", "yield"]

    data.to_csv(f"doyle_{name}.csv", index=False, header=True)


# %%
