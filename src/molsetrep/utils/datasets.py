from typing import List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import deepchem.molnet as mn
import pandas as pd


@dataclass
class CustomDataset:
    ids: np.array
    y: np.array

    @staticmethod
    def from_df(df, ids_column: str, y_columns: List[str]):
        return CustomDataset(
            df[ids_column].to_numpy(), df[df.columns.intersection(y_columns)].to_numpy()
        )


def adme_task_loader(name: str, freaturizer=None, **kwargs):
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
    valid = train  # train.sample(frac=0.1)

    tasks = ["activity"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def uspto_task_loader(name: str, freaturizer=None, **kwargs):
    return [
        "yield",
    ]


def uspto_loader(name: str, featurizer=None, seed=42, **kwargs):
    root_path = Path(__file__).resolve().parent
    uspto_path = Path(root_path, "../../../data/uspto/uspto_yields_above.csv.xz")

    df = pd.read_csv(uspto_path)

    train = df[df.split == "train"]
    test = df[df.split == "test"]

    # train = train.sample(frac=0.5)
    # test = test.sample(frac=0.5)

    # Validate on random sample from train
    valid = test.sample(frac=0.1)

    # valid, test = np.split(
    #     test.sample(frac=1.0, random_state=seed),
    #     [int(0.5 * len(test))],
    # )

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "rxn", tasks),
        CustomDataset.from_df(valid, "rxn", tasks),
        CustomDataset.from_df(test, "rxn", tasks),
        tasks,
        [],
    )


def suzuki_task_loader(name: str, freaturizer=None, **kwargs):
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
    df["smiles"] = df.smiles.str.replace("~", ".")

    train, test = np.split(
        df.sample(frac=1.0, random_state=seed),
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


def doyle_task_loader(name: str, freaturizer=None, **kwargs):
    return [
        "yield",
    ]


def doyle_loader(name: str, featurizer=None, split_ratio=0.7, seed=42, **kwargs):
    fold_idx = kwargs.get("fold_idx", 0)

    root_path = Path(__file__).resolve().parent
    doyle_path = Path(root_path, "../../../data/doyle")

    fold_files = []
    for file in doyle_path.glob("*.csv"):
        fold_files.append(file)

    df = pd.read_csv(fold_files[fold_idx])

    train, test = np.split(
        df.sample(frac=1.0, random_state=seed),
        [int(split_ratio * len(df))],
    )

    # Validate on random sample from train
    valid = df.sample(frac=0.1)

    tasks = ["yield"]

    return (
        CustomDataset.from_df(train, "smiles", tasks),
        CustomDataset.from_df(valid, "smiles", tasks),
        CustomDataset.from_df(test, "smiles", tasks),
        tasks,
        [],
    )


def ocelot_task_loader(name: str, freaturizer=None, **kwargs):
    root_path = Path(__file__).resolve().parent
    df = pd.read_csv(Path(root_path, "../../../data/ocelot_chromophore_v1.tar.xz"))
    col_names = list(df.columns)
    return col_names[1:-1]
    # return ["homo", "lumo"]


def ocelot_loader(name: str, featurizer=None, seed=42, **kwargs):
    root_path = Path(__file__).resolve().parent
    df = pd.read_csv(Path(root_path, "../../../data/ocelot_chromophore_v1.tar.xz"))
    col_names = list(df.columns)
    tasks = col_names[1:-1]
    # tasks = ["homo", "lumo"]

    train, valid, test = np.split(
        df.sample(frac=1.0, random_state=seed), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

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
