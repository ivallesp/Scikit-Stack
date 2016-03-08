# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'
import unittest

from sklearn.datasets import load_diabetes

from Stacker.data_operations import *
from demo.data import DIC_FOLD_PARTITIONS, DIC_SFOLD_PARTITIONS


class TestBasic(unittest.TestCase):
    def test_basic(self):
        diabetes = load_diabetes()
        X, y = diabetes["data"], diabetes["target"]
        self.assertEqual(len(X), len(y),
                         "Error loading diabetes data with sklearn. Train/test data with different sizes.")
        id = range(len(y))

    def test_seed_matching(self):
        # TEST KFOLD WITHOUT STRATIFYING
        cp = CrossPartitioner(k=10, n=100, stratify=False, shuffle=True, random_state=655321)
        gen = cp.make_partitions(kw=range(100), append_indices=False)
        for i, (train, test) in enumerate(gen):
            train_desired, test_desired = DIC_FOLD_PARTITIONS[i]
            self.assertTrue(len(test) == 0.1 * 100, "Bad test set partition size")
            self.assertTrue(len(train) == 0.9 * 100, "Bad train set partition size")
            self.assertTrue(len(train) + len(test) == 100, "The folds are not covering all the data instances")
            self.assertTrue(len(np.intersect1d(train, test)) == 0, "There are coincidences between train and test!!!")
            self.assertEqual(train.tolist(), train_desired)
            self.assertEqual(test.tolist(), test_desired)

        # TEST STRATIFIED K-FOLD
        cp = CrossPartitioner(k=10, y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, stratify=True, shuffle=True,
                              random_state=655321)
        gen = cp.make_partitions(kw=range(100), append_indices=False)
        for i, (train, test) in enumerate(gen):
            train_desired, test_desired = DIC_SFOLD_PARTITIONS[i]
            self.assertTrue(len(test) == 0.1 * 100, "Bad test set partition size")
            self.assertTrue(len(train) == 0.9 * 100, "Bad train set partition size")
            self.assertTrue(len(train) + len(test) == 100, "The folds are not covering all the data instances")
            self.assertTrue(len(np.intersect1d(train, test)) == 0, "There are coincidences between train and test!!!")
            self.assertEqual(train.tolist(), train_desired)
            self.assertEqual(test.tolist(), test_desired)
