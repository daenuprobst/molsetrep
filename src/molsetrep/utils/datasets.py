import deepchem.molnet as mn


def molnet_loader(name: str):
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader()

    tasks, dataset, lipo_transformers = dc_set
    train, valid, test = dataset
    return train, valid, test
