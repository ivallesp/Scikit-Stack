# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'

import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold


class CrossPartitioner():
    def __init__(self, n=None, y=None, k=10, stratify=False, shuffle=True, random_state=655321):
        """
        Function for creating the partitions for the skstack by using CrossValidation.
        :param n: When stratify=False, n defines the length of the datasets (int of None, default None)
        :param y: When stratify=True, train_y is the class used to preserve the percentages (array-like or None,
        default=None)
        :param k: Number of folds (int, default=10).
        :param stratify: Whether to preserve the percentage of samples of each class (boolean, default=False).
        :param shuffle: Whether to shuffle the data before splitting into batches (boolean, default=True).
        :param random_state: When shuffle=True, pseudo-random number generator state used for shuffling. If None, use
        default numpy RNG for shuffling (None, int or RandomState, default=655321)
        :return: None
        """
        if type(y) in (np.matrix, np.array):
            self.y = np.array(y)[:, 0] # TODO Add tests
        else:
            self.y = y
        self.k = k
        self.stratify = stratify
        self.shuffle = shuffle
        self.seed = random_state
        self.N = None
        if self.stratify:
            assert type(y) != None, "You must pass the  'train_y' parameter if you want to stratify."
            if n: assert len(
                y) == n, "The length of the parameter 'train_y' and the 'n' value don't mismatch. If you are " \
                         "stratifying, it is not necessary to specify n."
            self.cvIterator = StratifiedKFold(y = self.y,
                                              n_folds = self.k,
                                              shuffle = self.shuffle,
                                              random_state = self.seed)
            self.N = len(self.y)
        else:
            assert n != None, "You must specify the size of the data using the 'n' parameter if you don't stratify."
            self.cvIterator = KFold(n=n,
                                    n_folds = self.k,
                                    shuffle = self.shuffle,
                                    random_state = self.seed)
            self.N = n


    def make_partitions(self, append_indices = True, dict_format=False, **kwargs):
        """
        This function is feed by keyword arguments containing the data to be splitted in folds.
        :param append_indices: Returns the indices within the tuple containing the data split
        :param dict_format: Whether to return a dictionary. If False, it returns a list of tuples. (boolean,
        default=False)
        :param kwargs: keyword arguments which value is the data arrays to be partitioned. (array-like)
        :return: generator containing, for each data array, the data corresponding to the each fold in each case.
        Whether dict_format=True, it has de subsequent format:
        {
        "data1":    (train, test),
        "data2":    (train, test),
        ...
        }
        if dict_format=False, the format of the generator objects yielded are:
        [(train_d1, test_d1), (train_d2, test_d2), ...]
        """
        from scipy.sparse.csr import csr_matrix
        import pandas as pd
        for k, (train_index, test_index) in enumerate(self.cvIterator):
            partitioned_data = {} if dict_format else []

            for name, data in kwargs.items():
                if type(data) == list:
                    data = np.array(data)  # Converts lists to Numpy Array
                elif type(data) == pd.core.frame.DataFrame or type(data) == pd.core.series.Series:
                    data = np.array(data)


                if append_indices:
                    if type(data) == csr_matrix:
                        split = (data[train_index], data[test_index], train_index, test_index)  # Appends the indices
                    else:
                        split = (
                            data[[train_index]], data[[test_index]], train_index, test_index)  # Appends the indices
                else:
                    if type(data) == csr_matrix:
                        split = (data[train_index], data[test_index])  # Not appends the indices
                    else:
                        split = (data[[train_index]], data[[test_index]])  # Not appends the indices

                if dict_format:
                    partitioned_data[name] = split # Returns a dictionary
                else:
                    partitioned_data.append(split) # Returns a list
            if not dict_format and len(kwargs) == 1: partitioned_data = partitioned_data[0]
            yield partitioned_data
