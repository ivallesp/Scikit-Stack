__author__ = 'ivanvallesperez'
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Stacker.data_operations import CrossPartitioner


class Stacker():
    def __init__(self, train_X, train_y, train_id, folds=10, stratify=True):
        """
        This class is the main class of the Stacker. Its main purpose is to properly generate the stacked prediction
        assuring that the indices of the predictions are aligned.
        :param train_X: the input data for training (numpy.ndarray, scipy.sparse.csr, pandas.Dataframe)
        :param train_y: target for the training data (list, numpy.ndarray, pandas.Dataframe)
        :param train_id: id for the training data (list, numpy.ndarray, pandas.Dataframe). Used for assuring that the
        indices are properly aligned with the original data
        :param folds: Number of folds (int, default=10)
        :param stratify: Whether to preserve the percentage of samples of each class (boolean, default=False)
        :return: None
        """
        self.train_X = train_X
        self.train_y = train_y
        self.train_id = train_id
        self.model = None
        self.folds = folds
        self.stratify = stratify

    def generate_training_metapredictor(self, model):
        """
        This function is responsible for iterating across a CV loop with a specific (hard-coded) random seed, training
        the inner models and generating the stacked predictor (aligned with the training set).
        :param model: sklearn-like instanced model. It is requiered that it contains the 'fit' and 'predict_proba'
         methods (class).
        :return: pd.DataFrame with only one column containing the target. The ID of this DataFrame has to be aligned
        (i.e. to be the same) as the original training set index (pandas.Dataframe)
        """
        self.training_predictor = None
        self.test_predictor = None
        self.model = None
        self.model = model
        cp = CrossPartitioner(n=len(self.train_y) if not self.stratify else None,
                              y=self.train_y,
                              k=self.folds,
                              stratify=self.stratify,
                              shuffle=True,
                              random_state=655321)

        scores = []
        prediction_batches = []
        indices_batches = []
        gen = cp.make_partitions(input=self.train_X, target=self.train_y, ids=self.train_id, append_indices=False)
        for i, ((train_X_cv, test_X_cv), (train_y_cv, test_y_cv), (train_id_cv, test_id_cv)) in enumerate(gen):
            self.model.fit(train_X_cv, train_y_cv)
            test_prediction_cv = self.model.predict_proba(test_X_cv)  # Can give a 2D or a 1D Matrix
            test_prediction_cv = np.reshape(test_prediction_cv, (len(test_y_cv), test_prediction_cv.ndim))  # this code
            # forces having 2D
            test_prediction_cv = test_prediction_cv[:, -1]  # Extract the last column
            score = roc_auc_score(test_y_cv, test_prediction_cv)
            scores.append(score)
            assert len(test_id_cv) == len(test_prediction_cv)
            prediction_batches.extend(test_prediction_cv)
            indices_batches.extend(test_id_cv)
            assert len(prediction_batches) == len(indices_batches)
        self.cv_score_mean = np.mean(scores)
        self.cv_score_std = np.std(scores)
        training_predictor = pd.DataFrame({"target": prediction_batches}, index=indices_batches).ix[self.train_id]
        assert len(training_predictor) == len(self.train_X)
        self.training_predictor = training_predictor
        return training_predictor

    def generate_test_metapredictor(self, test_X, test_id):
        self.test_predictor = None
        self.model.fit(self.train_X, self.train_y)
        test_prediction = self.model.predict_proba(test_X)
        test_prediction = np.reshape(test_prediction, (len(test_id), test_prediction.ndim))  # this code
        # forces having 2D
        test_prediction = test_prediction[:, -1]  # Extract the last column
        test_predictor = pd.DataFrame({"target": test_prediction}, index=test_id)
        self.test_predictor = test_predictor
        return test_predictor

    def save_files(self, alias=None, folder="/tmp"):
        if not alias:
            training_filePath = os.path.join(folder, "training_metapredictor.csv")
            test_filePath = os.path.join(folder, "test_metapredictor.csv")
        else:
            training_filePath = os.path.join(folder, alias + "_training_metapredictor.csv")
            test_filePath = os.path.join(folder, alias + "_test_metapredictor.csv")
        self.training_predictor.to_csv(training_filePath, sep=",", encoding="utf-8", index_label="id")
        self.test_predictor.to_csv(test_filePath, sep=",", encoding="utf-8", index_label="id")
