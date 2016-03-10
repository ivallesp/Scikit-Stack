__author__ = 'ivanvallesperez'
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Stacker.data_operations import CrossPartitioner


class Stacker():
    def __init__(self, train_X, train_y, train_id, model, folds=10, stratify=True):
        self.train_X = train_X
        self.train_y = train_y
        self.train_id = train_id
        self.model = model
        self.folds = folds
        self.stratify = stratify

    def generate_training_metapredictor(self):  # TODO: Idea, move here the parameter "model"
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
        return training_predictor

    def generate_test_metapredictor(self, test_X, test_id):
        self.model.fit(self.train_X, self.train_y)
        test_prediction = self.model.predict_proba(test_X)
        test_prediction = np.reshape(test_prediction, (len(test_id), test_prediction.ndim))  # this code
        # forces having 2D
        test_prediction = test_prediction[:, -1]  # Extract the last column
        test_predictor = pd.DataFrame({"target": test_prediction}, index=test_id)
        return test_predictor
