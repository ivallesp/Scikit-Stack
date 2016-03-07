# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'
import unittest
from Stacker.data_operations import *
from sklearn.datasets import load_diabetes
from sklearn.cross_validation import train_test_split


class TestBasic(unittest.TestCase):
    def test_basic(self):
        diabetes = load_diabetes()
        X, y = diabetes["data"], diabetes["target"]
        self.assertEqual(len(X), len(y), "Error loading diabetes data with sklearn. Train/test data with different sizes.")
        id = range(len(y))

        cp = CrossPartitioner(k=10, y=y, stratify=True, schuffle=True, random_state=655321)
        cp = cp.fit(id)
        train_ids, test_ids = cp.transform(id)
        self.assertTrue(len(train_ids)==len(test_ids)==10, "Something went bad in the CrossPartitioner, got different"
                                                           " number of partitions than the number of folds defined.")



