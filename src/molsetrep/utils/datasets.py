import deepchem.molnet as mn


def molnet_loader(name: str, **kwargs):
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, dataset, lipo_transformers = dc_set
    train, valid, test = dataset
    return train, valid, test
