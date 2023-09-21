import numpy as np

from typing import Optional, Tuple
from deepchem.splits import Splitter
from deepchem.data.datasets import Dataset


class PDBBindLiSplitter(Splitter):
    """Class for doing random data splits.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> # Creating a dummy NumPy dataset
    >>> X, y = np.random.randn(5), np.random.randn(5)
    >>> dataset = dc.data.NumpyDataset(X, y)
    >>> # Creating a RandomSplitter object
    >>> splitter = dc.splits.RandomSplitter()
    >>> # Splitting dataset into train and test datasets
    >>> train_dataset, test_dataset = splitter.train_test_split(dataset)

    """

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.9,
        frac_valid: float = 0.1,
        frac_test: float = 0.0,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits internal compounds randomly into train/validation/test.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        seed: int, optional (default None)
            Random seed to use.
        frac_train: float, optional (default 0.9)
            The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
            The fraction of data to be used for the validation split.
        seed: int, optional (default None)
            Random seed to use.
        log_every_n: int, optional (default None)
            Log every n examples (not currently used).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple of train indices, valid indices, and test indices.
            Each indices is a numpy array.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        if seed is not None:
            np.random.seed(seed)
        num_datapoints = len(dataset)
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
        shuffled = np.random.permutation(range(num_datapoints))
        return (
            shuffled[:train_cutoff],
            shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:],
        )
