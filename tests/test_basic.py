# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'

import unittest

import scipy as sp
from sklearn.datasets import *

from Stacker.stacker import *
from demo.data import DIC_FOLD_PARTITIONS, DIC_SFOLD_PARTITIONS


class TestBasic(unittest.TestCase):
    def test_basic(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        model = RandomForestClassifier(n_estimators=100, random_state=655321)
        data = make_hastie_10_2(2000)
        index_randperm = np.random.permutation(range(2000))
        data = [data[0][index_randperm, :], data[1][index_randperm]]
        out_of_sample = [data[0][1000:2000, :], data[1][1000:2000]]
        data = [data[0][0:1000, :], data[1][0:1000]]
        test_X, test_y = out_of_sample[0], out_of_sample[1]
        X, y = data[0], data[1]
        id = np.random.permutation(range(len(y)))  # We random permutate it to make the problem harder.
        # I am very worried to mess the indices...
        test_id = np.random.permutation(range(len(test_y)))
        self.assertEqual(len(X), len(y),
                         "Error loading hastie data with sklearn. Train/test data with different sizes.")

        s = Stacker(train_X=X, train_y=y, train_id=id, stratify=False)
        y_hat_training = s.generate_training_metapredictor(model=model)
        self.assertIn("cv_score_mean", dir(s), "Atribute 'cv_score_mean' not found in the Stacker object")
        self.assertIn("cv_score_std", dir(s), "Atribute 'cv_score_std' not found in the Stacker object")
        self.assertAlmostEqual(s.cv_score_mean, 0.9, delta=0.05, msg="Score value given by the model ('%s' ± '%s') "
                                                                     "is weird." % (s.cv_score_mean, s.cv_score_std))
        self.assertAlmostEqual(roc_auc_score(y, y_hat_training), s.cv_score_mean, places=2)
        # Differences produced basically because the size of the sample when calculating the AUC gives the resolution of
        # the curve. As the resolution changes, the result also slightly changes. We are calculating the roc with
        # different sample sizes.
        self.assertIn("training_predictor", dir(s), "Atribute 'training_predictor' not found in the Stacker object")
        self.assertTrue(type(s.training_predictor) != None)

        y_hat_test = s.generate_test_metapredictor(test_X, test_id)
        self.assertAlmostEqual(roc_auc_score(test_y, y_hat_test), s.cv_score_mean, delta=0.05)
        # Test score greater than test_cv score because first of all I am not making any fit, so the cv out of sample
        # set is completely out of sample, and for the test set I am using all the training data, what produces a
        # harder training
        self.assertIn("test_predictor", dir(s), "Atribute 'test_predictor' not found in the Stacker object")
        self.assertTrue(type(s.test_predictor) != None)

    def test_consistencies(self):
        """
        In this test, let's check if the predictors are bad generated if we instantiate the class and try to build 2
        different predictors using 2 different models. We are going to check also if the indices of the predictors
        generated are aligned. We also will assert that the correlation between predictions is high.
        """
        from sklearn.ensemble import RandomForestClassifier

        data = make_hastie_10_2(2000)
        index_randperm = np.random.permutation(range(2000))
        data = [data[0][index_randperm, :], data[1][index_randperm]]
        out_of_sample = [data[0][1000:2000, :], data[1][1000:2000]]
        data = [data[0][0:1000, :], data[1][0:1000]]
        test_X, test_y = out_of_sample[0], out_of_sample[1]
        X, y = data[0], data[1]
        id = np.random.permutation(range(len(y)))  # We random permutate it to make the problem harder.
        # I am very worried to mess the indices...
        test_id = np.random.permutation(range(len(test_y)))
        self.assertEqual(len(X), len(y),
                         "Error loading hastie data with sklearn. Train/test data with different sizes.")
        s = Stacker(train_X=X, train_y=y, train_id=id, stratify=False)

        model = RandomForestClassifier(n_estimators=100, random_state=655321)
        y_hat_training = s.generate_training_metapredictor(model=model)
        y_hat_test = s.generate_test_metapredictor(test_X, test_id)

        # Regenerate the predictors to check if there are differences
        self.assertTrue(y_hat_training.equals(s.generate_training_metapredictor(model=model)))
        self.assertTrue(y_hat_test.equals(s.generate_test_metapredictor(test_X, test_id)))

        model = RandomForestClassifier(n_estimators=100, random_state=112358)
        y_hat_training_2 = s.generate_training_metapredictor(model=model)
        y_hat_test_2 = s.generate_test_metapredictor(test_X, test_id)

        # Regenerate the predictors to check if there are differences
        self.assertTrue(y_hat_training_2.equals(s.generate_training_metapredictor(model=model)))
        self.assertTrue(y_hat_test_2.equals(s.generate_test_metapredictor(test_X, test_id)))

        # Predictors generated are different (obvious but necessary)
        self.assertFalse(y_hat_training.equals(y_hat_training_2))
        self.assertFalse(y_hat_test.equals(y_hat_test_2))

        # Check the index alignments
        self.assertEqual(y_hat_test.index.tolist(), test_id.tolist())
        self.assertEqual(y_hat_training.index.tolist(), id.tolist())
        self.assertEqual(y_hat_test.index.tolist(), y_hat_test_2.index.tolist())
        self.assertEqual(y_hat_training.index.tolist(), y_hat_training_2.index.tolist())
        self.assertNotEqual(y_hat_test.index.tolist(), y_hat_training_2.index.tolist())
        self.assertNotEqual(y_hat_training.index.tolist(), y_hat_test_2.index.tolist())

        # Check the results
        self.assertAlmostEqual(roc_auc_score(y, y_hat_training), 0.9, delta=0.05)
        self.assertAlmostEqual(roc_auc_score(test_y, y_hat_test), 0.9, delta=0.05)
        self.assertAlmostEqual(roc_auc_score(y, y_hat_training_2), 0.9, delta=0.05)
        self.assertAlmostEqual(roc_auc_score(test_y, y_hat_test_2), 0.9, delta=0.05)

        self.assertFalse(y_hat_training.equals(y_hat_training_2))
        self.assertFalse(y_hat_test.equals(y_hat_test_2))
        # The correlation between both model predictions has to be high because we are training the same model but
        # with different seed
        self.assertGreater(sp.stats.stats.pearsonr(y_hat_training, y_hat_training_2)[0][0], 0.9)
        self.assertGreater(sp.stats.stats.pearsonr(y_hat_training, y_hat_training_2)[0][0], 0.9)

    def test_file_storing(self):
        from sklearn.ensemble import RandomForestClassifier

        data = make_hastie_10_2(2000)
        index_randperm = np.random.permutation(range(2000))
        data = [data[0][index_randperm, :], data[1][index_randperm]]
        out_of_sample = [data[0][1000:2000, :], data[1][1000:2000]]
        data = [data[0][0:1000, :], data[1][0:1000]]
        test_X, test_y = out_of_sample[0], out_of_sample[1]
        X, y = data[0], data[1]
        id = np.random.permutation(range(len(y)))  # We random permutate it to make the problem harder.
        # I am very worried to mess the indices...
        test_id = np.random.permutation(range(len(test_y)))
        self.assertEqual(len(X), len(y),
                         "Error loading hastie data with sklearn. Train/test data with different sizes.")
        s = Stacker(train_X=X, train_y=y, train_id=id, stratify=False)

        model = RandomForestClassifier(n_estimators=100, random_state=655321)
        y_hat_training = s.generate_training_metapredictor(model=model)
        y_hat_test = s.generate_test_metapredictor(test_X, test_id)

        s.save_files(alias="demo_predictor", folder="../demo")
        pathTraining = "../demo/demo_predictor_training_metapredictor.csv"
        pathTest = "../demo/demo_predictor_test_metapredictor.csv"
        self.assertTrue(os.path.exists(pathTraining))
        self.assertTrue(os.path.exists(pathTest))
        dfTrain = pd.read_csv(pathTraining, sep=",", encoding="utf-8", index_col="id")
        dfTest = pd.read_csv(pathTest, sep=",", encoding="utf-8", index_col="id")
        self.assertTrue(len(dfTrain.columns) == 1)
        self.assertTrue(len(dfTest.columns) == 1)
        self.assertEqual(dfTrain.index.tolist(), id.tolist())
        self.assertEqual(dfTest.index.tolist(), test_id.tolist())
        self.assertIn("target", dfTrain.columns)
        self.assertIn("target", dfTest.columns)
        os.remove(pathTraining)
        os.remove(pathTest)


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

        # TEST BOTH TOGETHER (test with 2 datasets)
        cp1 = CrossPartitioner(k=10, n=100, stratify=False, shuffle=True, random_state=655321)
        gen1 = cp1.make_partitions(kw=range(100), append_indices=False)
        cp2 = CrossPartitioner(k=10, y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, stratify=True, shuffle=True,
                               random_state=655321)
        gen2 = cp2.make_partitions(kw=range(100), append_indices=False)
        for i, ((train1, test1), (train2, test2)) in enumerate((zip(gen1, gen2))):
            train_desired1, test_desired1 = DIC_FOLD_PARTITIONS[i]
            self.assertTrue(len(test1) == 0.1 * 100, "Bad test set partition size")
            self.assertTrue(len(train1) == 0.9 * 100, "Bad train set partition size")
            self.assertTrue(len(train1) + len(test1) == 100, "The folds are not covering all the data instances")
            self.assertTrue(len(np.intersect1d(train1, test1)) == 0, "There are coincidences between train and test!!!")
            self.assertEqual(train1.tolist(), train_desired1)
            self.assertEqual(test1.tolist(), test_desired1)
            train_desired2, test_desired2 = DIC_SFOLD_PARTITIONS[i]
            self.assertTrue(len(test2) == 0.1 * 100, "Bad test set partition size")
            self.assertTrue(len(train2) == 0.9 * 100, "Bad train set partition size")
            self.assertTrue(len(train2) + len(test2) == 100, "The folds are not covering all the data instances")
            self.assertTrue(len(np.intersect1d(train2, test2)) == 0, "There are coincidences between train and test!!!")
            self.assertEqual(train2.tolist(), train_desired2)
            self.assertEqual(test2.tolist(), test_desired2)

    def test_numpy_dense_datatype(self):
        data = np.transpose(np.array([range(100), range(100, 200), range(200, 300)]))
        cp = CrossPartitioner(k=10, n=100, stratify=False, shuffle=True, random_state=655321)
        gen = cp.make_partitions(kw=data, append_indices=False)
        # TEST K-FOLD
        for i, (train, test) in enumerate(gen):
            self.assertTrue(train.shape[1] == test.shape[1] == 3, "Different data columns returned!!")
            self.assertEqual(train.shape[0] + test.shape[0], 100, "Different number of total rows returned!!")
            self.assertTrue(train.shape[0] > test.shape[0], "Test size greater than train size!!")

        # TEST STRATIFIED K-FOLD
        cp = CrossPartitioner(k=10, y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, stratify=True, shuffle=True,
                              random_state=655321)
        gen = cp.make_partitions(kw=data, append_indices=False)
        for i, (train, test) in enumerate(gen):
            self.assertTrue(train.shape[1] == test.shape[1] == 3, "Different data columns returned!!")
            self.assertEqual(train.shape[0] + test.shape[0], 100, "Different number of total rows returned!!")
            self.assertTrue(train.shape[0] > test.shape[0], "Test size greater than train size!!")

    def test_numpy_csr_datatype(self):
        from scipy.sparse import csr_matrix
        data = csr_matrix(np.transpose(np.array([range(100), range(100, 200), range(200, 300)])))
        # TEST K-FOLD
        cp = CrossPartitioner(k=10, n=100, stratify=False, shuffle=True, random_state=655321)
        gen = cp.make_partitions(kw=data, append_indices=False)
        for i, (train, test) in enumerate(gen):
            self.assertTrue(train.shape[1] == test.shape[1] == 3, "Different data columns returned!!")
            self.assertEqual(train.shape[0] + test.shape[0], 100, "Different number of total rows returned!!")
            self.assertTrue(train.shape[0] > test.shape[0], "Test size greater than train size!!")

        # TEST STRATIFIED K-FOLD
        cp = CrossPartitioner(k=10, y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, stratify=True, shuffle=True,
                              random_state=655321)
        gen = cp.make_partitions(kw=data, append_indices=False)
        for i, (train, test) in enumerate(gen):
            self.assertTrue(train.shape[1] == test.shape[1] == 3, "Different data columns returned!!")
            self.assertEqual(train.shape[0] + test.shape[0], 100, "Different number of total rows returned!!")
            self.assertTrue(train.shape[0] > test.shape[0], "Test size greater than train size!!")

    def test_pandas_dataframe_datatype(self):
        import pandas as pd
        data = pd.DataFrame({"uno": range(100), "dos": range(100, 200), "tres": range(200, 300)})
        # TEST K-FOLD
        cp = CrossPartitioner(k=10, n=100, stratify=False, shuffle=True, random_state=655321)
        gen = cp.make_partitions(kw=data, append_indices=False)
        for i, (train, test) in enumerate(gen):
            self.assertTrue(train.shape[1] == test.shape[1] == 3, "Different data columns returned!!")
            self.assertEqual(train.shape[0] + test.shape[0], 100, "Different number of total rows returned!!")
            self.assertTrue(train.shape[0] > test.shape[0], "Test size greater than train size!!")
        # TEST STRATIFIED K-FOLD
        cp = CrossPartitioner(k=10, y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, stratify=True, shuffle=True,
                              random_state=655321)
        gen = cp.make_partitions(kw=data, append_indices=False)
        for i, (train, test) in enumerate(gen):
            self.assertTrue(train.shape[1] == test.shape[1] == 3, "Different data columns returned!!")
            self.assertEqual(train.shape[0] + test.shape[0], 100, "Different number of total rows returned!!")
            self.assertTrue(train.shape[0] > test.shape[0], "Test size greater than train size!!")
