__author__ = 'ivanvallesperez'
import numpy as np
from sklearn.metrics import roc_auc_score

from Stacker.data_operations import CrossPartitioner


class Stacker():
    def __init__(self, train_X, train_y, train_id, model, folds=10, stratify=True):
        self.train_x = train_X
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

        gen = cp.make_partitions(input=self.train_x, target=self.train_y, ids=self.train_id, append_indices=False)
        for i, ((inner_train_X, inner_test_X), (inner_train_y, inner_test_y),
                (inner_train_id, inner_test_id)) in enumerate(gen):
            print inner_train_X.shape, inner_test_X.shape
            self.model.fit(inner_train_X, inner_train_y)
            inner_test_prediction = self.model.predict_proba(inner_test_X)  # Can give a 2D or a 1D Matrix
            inner_test_prediction = np.reshape(inner_test_prediction, (
                len(inner_test_y), inner_test_prediction.ndim))  # this code forces having 2D
            inner_test_prediction = inner_test_prediction[:, -1]  # Procudes the last column

            print roc_auc_score(inner_test_y, inner_test_prediction)
