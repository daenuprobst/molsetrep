import pickle
from typing import List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import deepchem.molnet as mn
import pandas as pd

from deepchem.splits import RandomSplitter
from deepchem.utils.rdkit_utils import load_molecule
from sklearn.model_selection import KFold

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


@dataclass
class CustomDataset:
    ids: np.ndarray
    y: np.ndarray

    @staticmethod
    def from_df(df, ids_column: str, y_columns: List[str]):
        return CustomDataset(
            df[ids_column].to_numpy(), df[df.columns.intersection(y_columns)].to_numpy()
        )


@dataclass
class CustomPDBBindDataset:
    X: np.ndarray
    y: np.ndarray

    @staticmethod
    def from_df(X: np.ndarray, y: np.ndarray):
        return CustomPDBBindDataset(X, y)


##############################


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, row in dataset.iterrows():
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))

        scaffold = _generate_scaffold(row["smiles"])

        # Adapted from original to account for SMILES not readable by MolFromSmiles
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


##############################

molnet_tasks = {
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "esol": ["ESOL predicted log solubility in mols per litre"],
    "freesolv": ["expt"],
    "hiv": ["HIV_active"],
    "lipo": ["exp"],
    "muv": [
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    ],
    "qm7": ["u0_atom"],
    "qm8": [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ],
    "qm9": ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"],
    "sider": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}


def pdbbind_custom_task_loader(name: str, featurizer=None, **kwargs):
    return ["-logKd/Ki"]


def pdbbind_custom_loader(
    name: str, featurizer=None, split_ratio=0.7, seed=42, task_name=None, **kwargs
):
    root_path = Path(__file__).resolve().parent
    meta_path = Path(root_path, "../../../data/pdbbind/meta.csv")

    data = {"train": [[], []], "valid": [[], []], "test": [[], []]}
    df = pd.read_csv(meta_path)

    for _, row in df.iterrows():
        data[row["split"]][0].append(
            (
                load_molecule(row["mol_path"], calc_charges=False, add_hydrogens=False)[
                    1
                ],
                load_molecule(
                    row["pocket_path"], calc_charges=False, add_hydrogens=False
                )[1],
            )
        )
        data[row["split"]][1].append([row["label"]])

    train = CustomPDBBindDataset(data["train"][0], np.array(data["train"][1]))
    valid = CustomPDBBindDataset(data["valid"][0], np.array(data["valid"][1]))
    test = CustomPDBBindDataset(data["test"][0], np.array(data["test"][1]))

    return train, valid, test, ["-logKd/Ki"], None


def custom_molnet_task_loader(name: str, featurizer=None, **kwargs):
    return molnet_tasks[name]


def custom_molnet_loader(
    name: str, featurizer=None, split_ratio=0.7, seed=42, task_name=None, **kwargs
):
    root_path = Path(__file__).resolve().parent
    file_path = Path(root_path, f"../../../data/moleculenet/{name}.csv.xz")

    df = pd.read_csv(file_path)

    # Drop NAs. Needed in Tox21
    if name in ["tox21"]:
        df = df.replace("", np.nan)
        df = df.dropna(subset=[task_name])

    train_ids, valid_ids, test_ids = scaffold_split(df, 0.1, 0.1, seed)

    train = df.loc[train_ids]
    valid = df.loc[valid_ids]
    test = df.loc[test_ids]

    tasks = molnet_tasks[name]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def adme_task_loader(name: str, featurizer=None, **kwargs):
    return ["HLM", "hPPB", "MDR1_ER", "RLM", "rPPB", "Sol"]


def adme_loader(name: str, featurizer=None, split_ratio=0.7, seed=42, **kwargs):
    task_name = kwargs.get("task_name", None)
    print(task_name)

    root_path = Path(__file__).resolve().parent
    adme_train_file = Path(root_path, f"../../../data/adme/ADME_{task_name}_train.csv")
    adme_test_file = Path(root_path, f"../../../data/adme/ADME_{task_name}_test.csv")

    train = pd.read_csv(adme_train_file)
    test = pd.read_csv(adme_test_file)

    # Validate on random sample from train
    valid = train.sample(frac=0.1)

    tasks = ["activity"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def uspto_task_loader(name: str, featurizer=None, **kwargs):
    return [
        "yield",
    ]


def uspto_loader(name: str, featurizer=None, seed=42, **kwargs):
    root_path = Path(__file__).resolve().parent
    uspto_path = Path(root_path, "../../../data/uspto/uspto_yields_above.csv.xz")

    df = pd.read_csv(uspto_path)

    train = df[df.split == "train"]
    test = df[df.split == "test"]

    # Validate on random sample from train
    valid = train.sample(frac=0.1)

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "rxn", tasks),
        CustomDataset.from_df(valid, "rxn", tasks),
        CustomDataset.from_df(test, "rxn", tasks),
        tasks,
        [],
    )


def az_task_loader(name: str, featurizer=None, **kwargs):
    return [
        "yield",
    ]


def az_loader(name: str, featurizer=None, split_ratio=0.7, seed=42, **kwargs):
    fold_idx = kwargs.get("fold_idx", 0)

    root_path = Path(__file__).resolve().parent
    az_path = Path(root_path, "../../../data/az")
    splits = pickle.load(open(Path(az_path, "train_test_idxs.pickle"), "rb"))

    train_ids = splits["train_idx"][fold_idx + 1]
    test_ids = splits["test_idx"][fold_idx + 1]

    df = pd.read_csv(Path(az_path, "az_no_rdkit.csv"))
    df["smiles"] = (
        df.reactant_smiles
        + "."
        + df.solvent_smiles
        + "."
        + df.base_smiles
        + ">>"
        + df.product_smiles
    )

    train = df.iloc[train_ids]
    test = df.iloc[test_ids]

    # Validate on random sample from train
    valid = train.sample(frac=0.1)

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def suzuki_task_loader(name: str, featurizer=None, **kwargs):
    return [
        "yield",
    ]


def suzuki_loader(name: str, featurizer=None, split_ratio=0.7, seed=42, **kwargs):
    fold_idx = kwargs.get("fold_idx", 0)

    root_path = Path(__file__).resolve().parent
    suzuki_path = Path(root_path, "../../../data/suzuki_miyaura")

    fold_files = []
    for file in suzuki_path.glob("*.csv"):
        fold_files.append(file)

    df = pd.read_csv(fold_files[fold_idx])
    df["smiles"] = df.smiles.str.replace("~", "")

    train, test = np.split(
        df,
        [int(split_ratio * len(df))],
    )

    # Validate on random sample from train
    # valid = df.sample(frac=0.1)

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def doyle_test_loader(name: str, featurizer=None, split_ratio=0.7, seed=42, **kwargs):
    fold_idx = kwargs.get("fold_idx", 0)
    fold_idx = fold_idx % 4

    root_path = Path(__file__).resolve().parent
    doyle_path = Path(root_path, "../../../data/doyle")

    df = pd.read_csv(Path(doyle_path, f"doyle_Test{fold_idx + 1}.csv"))

    train, test = np.split(
        df,
        [3058],
    )

    # Validate on random sample from train
    valid = train.sample(frac=0.1)

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def doyle_task_loader(name: str, featurizer=None, **kwargs):
    return [
        "yield",
    ]


def doyle_loader(name: str, featurizer=None, split_ratio=0.7, seed=42, **kwargs):
    fold_idx = kwargs.get("fold_idx", 0)

    root_path = Path(__file__).resolve().parent
    doyle_path = Path(root_path, "../../../data/doyle")

    fold_files = []
    for file in doyle_path.glob("*FullCV*.csv"):
        fold_files.append(file)

    df = pd.read_csv(fold_files[fold_idx])

    train, test = np.split(
        df,
        [int(split_ratio * len(df))],
    )

    # Validate on random sample from train
    valid = train.sample(frac=0.1)
    if len(valid) < 16:
        valid = train.sample(n=16)

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def ocelot_task_loader(name: str, featurizer=None, **kwargs):
    root_path = Path(__file__).resolve().parent
    df = pd.read_csv(Path(root_path, "../../../data/ocelot_chromophore_v1.tar.xz"))
    col_names = list(df.columns)
    return col_names[1:-1]


def ocelot_loader(name: str, featurizer=None, seed=42, **kwargs):
    fold_idx = kwargs.get("fold_idx", 0)
    root_path = Path(__file__).resolve().parent
    df = pd.read_csv(Path(root_path, "../../../data/ocelot_chromophore_v1.tar.xz"))
    col_names = list(df.columns)
    tasks = col_names[1:-1]

    # For some reason one smiles entry is nan when running on colab
    df = df.dropna()

    # Keep the same random state
    kf = KFold(n_splits=5)
    train_ids, test_ids = list(kf.split(df))[fold_idx]

    train = df.iloc[train_ids]
    test = df.iloc[test_ids]

    valid = train.sample(frac=0.2)

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def molnet_task_loader(name: str, featurizer=None, **kwargs):
    mn_loader = getattr(mn, f"load_{name}")

    if featurizer:
        dc_set = mn_loader(featurizer=featurizer, **kwargs)
    else:
        dc_set = mn_loader(**kwargs)

    tasks, _, _ = dc_set
    return tasks


def molnet_loader(name: str, featurizer=None, **kwargs):
    mn_loader = getattr(mn, f"load_{name}")
    if featurizer:
        dc_set = mn_loader(featurizer=featurizer, **kwargs)
    else:
        dc_set = mn_loader(**kwargs)

    tasks, dataset, transformers = dc_set
    train, valid, test = dataset
    return train, valid, test, tasks, transformers


def get_class_weights(y, task_idx=None):
    if task_idx is None:
        _, counts = np.unique(y, return_counts=True)
        weights = [1 - c / y.shape[0] for c in counts]

        return np.array(weights), np.array(counts)
    else:
        y_t = y.T

        _, counts = np.unique(y_t[task_idx], return_counts=True)
        weights = [1 - c / y_t[task_idx].shape[0] for c in counts]

        return np.array(weights), np.array(counts)
