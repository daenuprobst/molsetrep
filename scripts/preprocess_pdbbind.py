import sys
import random
from pathlib import Path
from shutil import rmtree
import typer
import pandas as pd


def main(refined_folder: str, core_folder: str, index_file: str, out_file: str):
    print(
        f"Continuing will remove samples found in '{core_folder}' from '{refined_folder}'."
    )
    print("Do you *really* want to do that? (y/n)")

    answer = ""

    while answer not in ["y", "n"]:
        answer = input("Answer: ")

    if answer == "n":
        sys.exit()

    refined_samples = [f.name for f in Path(refined_folder).glob("*")]
    core_samples = [f.name for f in Path(core_folder).glob("*")]
    intersecting_samples = list(set(refined_samples) & set(core_samples))

    for sample in intersecting_samples:
        sample_path = Path(Path(refined_folder), sample)
        rmtree(str(sample_path))

        print(f"Deleted '{sample_path}'")

    df = pd.read_csv(
        index_file,
        sep=r"\s+",
        comment="#",
        header=None,
    )

    name_label_map = {}
    for _, row in df.iterrows():
        name_label_map[row[0]] = row[3]

    test_items = []
    for f in Path(core_folder).glob("*"):
        name = f.name
        label = name_label_map[name]
        test_items.append(
            {
                "name": name,
                "label": label,
                "mol_path": str(
                    Path(core_folder, name, f"{name}_ligand_opt.mol2").resolve()
                ),
                "pocket_path": str(
                    Path(core_folder, name, f"{name}_pocket.pdb").resolve()
                ),
                "split": "test",
            }
        )

    refined_items = []
    for f in Path(refined_folder).glob("*"):
        if f.name in ["index", "readme"]:
            continue

        name = f.name
        label = name_label_map[name]
        refined_items.append(
            {
                "name": name,
                "label": label,
                "mol_path": str(
                    Path(refined_folder, name, f"{name}_ligand.mol2").resolve()
                ),
                "pocket_path": str(
                    Path(refined_folder, name, f"{name}_pocket.pdb").resolve()
                ),
                "split": "train",
            }
        )

    random.seed(666)
    random.shuffle(refined_items)

    split_idx = int(round(len(refined_items) * 0.9))
    train_items = refined_items[:split_idx]
    valid_items = refined_items[split_idx:]

    for item in valid_items:
        item.update({"split": "valid"})

    if not out_file.endswith(".csv"):
        out_file += ".csv"

    df_out = pd.DataFrame.from_dict(train_items + valid_items + test_items)
    df_out.to_csv(out_file, index=False)


if __name__ == "__main__":
    typer.run(main)
