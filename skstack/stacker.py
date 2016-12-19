__author__ = 'ivanvallesperez'
import codecs
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error

from skstack.common_objects import ParameterFail
from skstack.data_operations import CrossPartitioner
from skstack.file_tools import make_sure_do_not_replace

class Stacker():
    def __init__(self, train_X, train_y, train_id, folds=10, stratify=True, cv_grouping=None, metric="auc"):
        """
        This class is the main class of the skstack. Its main purpose is to properly generate the stacked prediction
        assuring that the indices of the predictions are aligned.
        :param train_X: the input data for training (numpy.ndarray, scipy.sparse.csr, pandas.Dataframe)
        :param train_y: target for the training data (list, numpy.ndarray, pandas.Dataframe)
        :param train_id: id for the training data (list, numpy.ndarray, pandas.Dataframe). Used for assuring that the
        indices are properly aligned with the original data
        :param folds: Number of folds (int, default=10)
        :param stratify: Whether to preserve the percentage of samples of each class (boolean, default=False)
        :param cv_grouping: Whether to use an alternative index for stratifying the folds (list or None, default=None)
        :param metric: 'auc', 'logloss', 'mae', None or a custom metric with the format func(y, prob)
        :return: None
        """
        self.train_X = train_X
        self.train_y = train_y
        self.train_id = train_id
        self.folds = folds
        if "tolist" in dir(cv_grouping): cv_grouping = cv_grouping.tolist()
        self.stratify = stratify if not cv_grouping else True
        self.cv_grouping = self.train_y if not cv_grouping else cv_grouping
        self.metric = metric
        self.is_whole_model_trained = None

    def generate_training_metapredictor(self, model):
        """
        This function is responsible for iterating across a CV loop with a specific (hard-coded) random seed, training
        the inner models and generating the stacked predictor (aligned with the training set).
        :param model: sklearn-like instanced model. It is requiered that it contains the 'fit' and 'predict'
         methods. If it contains a 'predict_proba' method, the 'predict' method will be replaced by the 'predict_proba'
          one in this method (class).
        :return: pd.DataFrame with only one column containing the target. The ID of this DataFrame has to be aligned
        (i.e. to be the same) as the original training set index (pandas.Dataframe)
        """
        self.training_predictor = None
        self.test_predictor = None
        self.model = None
        self.model = model
        self.is_whole_model_trained = False

        if "predict_proba" in dir(self.model): self.model.predict = self.model.predict_proba

        if self.metric == "auc":
            eval_metric = roc_auc_score
        elif self.metric == "logloss":
            eval_metric = log_loss
        elif self.metric == "mae":
            eval_metric = mean_absolute_error
        else:
            eval_metric = self.metric  # Custom eval metric or None

        cp = CrossPartitioner(n=len(self.train_y) if not self.stratify else None,
                              y=self.cv_grouping,
                              k=self.folds,
                              stratify=self.stratify,
                              shuffle=True,
                              random_state=655321)

        scores = []
        prediction_batches = []
        indices_batches = []
        t1 = time.time()
        gen = cp.make_partitions(input=self.train_X, target=self.train_y, ids=self.train_id, append_indices=False)
        for i, ((train_X_cv, test_X_cv), (train_y_cv, test_y_cv), (train_id_cv, test_id_cv)) in enumerate(gen):
            self.model.fit(train_X_cv, train_y_cv)
            test_prediction_cv = self.model.predict(test_X_cv)  # Can give a 2D or a 1D Matrix
            # TODO: Add Tests if needed
            if len(test_prediction_cv.shape) > 1:
                if test_prediction_cv.shape[0] < test_prediction_cv.shape[1]: # Forces vertical table
                    test_prediction_cv = test_prediction_cv.T
                test_prediction_cv = test_prediction_cv[:, -1]  # Extract the last column
            score = eval_metric(list(np.array(test_y_cv)), list(np.array(test_prediction_cv))) if eval_metric else None
            scores.append(score)
            assert len(test_id_cv) == len(test_prediction_cv)
            prediction_batches.extend(test_prediction_cv)
            indices_batches.extend(test_id_cv)
            assert len(prediction_batches) == len(indices_batches)
        t2 = time.time()
        self.cv_time = t2 - t1
        self.cv_score_mean = np.mean(scores) if eval_metric else None
        self.cv_score_std = np.std(scores) if eval_metric else None
        training_predictor = pd.DataFrame({"target": prediction_batches}, index=indices_batches).ix[self.train_id]
        assert len(training_predictor) == len(self.train_X)
        self.training_predictor = training_predictor
        return training_predictor

    def generate_test_metapredictor(self, test_X, test_id):
        """
        This function is responsible for generating the test metapredictor. For this, it trains the model stored in the
        "generate_train_metapredictor" method with all the training data (also specified in the previous method).
        :param test_X: Test set (numpy.ndarray, scipy.sparse.csr, pandas.Dataframe)
        :param test_id: id for the test data (list, numpy.ndarray, pandas.Dataframe). Used for assuring that the
        indices are properly aligned with the original data
        :return: pd.DataFrame with only one column containing the target. The ID of this DataFrame has to be aligned
        (i.e. to be the same) as the original test set index (pandas.Dataframe)
        """
        self.test_predictor = None
        t1 = time.time()
        if not self.is_whole_model_trained:
            self.model.fit(self.train_X, self.train_y)
            self.is_whole_model_trained = True
        test_prediction = self.model.predict(test_X)
        t2 = time.time()
        self.whole_training_time = t2 - t1
        if len(test_prediction.shape) > 1:
            if test_prediction.shape[0] < test_prediction.shape[1]:  # Forces vertical table
                test_prediction = test_prediction.T
        #test_prediction = np.reshape(test_prediction, (len(test_id), test_prediction.ndim))  # this code
        # forces having 2D
            test_prediction = test_prediction[:, -1]  # Extract the last column
        test_predictor = pd.DataFrame({"target": test_prediction}, index=test_id)
        self.test_predictor = test_predictor
        return test_predictor

    def save_files(self, alias=None, folder="/tmp", metadata=None):
        """
        This method is intended for creating the predictor files and an index file for gathering all the stats.
        :param alias: Name of the file to be created (str or unicode)
        :param folder: Path to the folder that will contain all the predictors (str or unicode)
        :param metadata: Dictionary containing the keys "name" and "description". It can contain some optional keys,
        but they have to be convertible to json with the method json.dumps() of the library json
        :return: None
        """
        if not metadata:
            raise ParameterFail("Metadata parameter not defined!")
        else:
            if not "name" in metadata or not "description" in metadata:
                raise ParameterFail("Name and/or description keys not found in metadata")

        if not alias:
            training_filePath = os.path.join(folder, "training_metapredictor.csv")
            test_filePath = os.path.join(folder, "test_metapredictor.csv")
        else:
            training_filePath = os.path.join(folder, alias + "_training_metapredictor.csv")
            test_filePath = os.path.join(folder, alias + "_test_metapredictor.csv")

        training_filePath = make_sure_do_not_replace(training_filePath)
        test_filePath = make_sure_do_not_replace(test_filePath)

        self.training_predictor.to_csv(training_filePath, sep=",", encoding="utf-8", index_label="id")
        self.test_predictor.to_csv(test_filePath, sep=",", encoding="utf-8", index_label="id")

        index = {
            "name": metadata["name"],
            "description": metadata["description"],
            "cv": {
                "score_mean": self.cv_score_mean,
                "score_std": self.cv_score_std,
                "score_metric": self.metric,
                "folds": self.folds,
                "stratify": self.stratify,
                "shuffle": True,
                "time": self.cv_time
            },
            "whole_model_time": self.whole_training_time,
            "total_time": self.whole_training_time + self.cv_time,
            "alias": alias,
            "test_filePath": test_filePath,
            "train_filePath": training_filePath,
            "datetime": datetime.datetime.now().isoformat(),
        }

        for key in metadata:
            if key == "name" or key == "description" or key in index:
                continue
            else:
                index[key] = metadata[key]
        json_dump = json.dumps(index)
        with codecs.open(os.path.join(folder, "index.jl"), "a", "utf-8") as f:
            f.write(json_dump + "\r\n")
