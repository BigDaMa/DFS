import numpy as np
from scipy.sparse import vstack


class RecordError(): #n_records_for_each_class

    def __init__(self, stat_model):
        self.stat_model = stat_model


    def get_reward(self, x, y, record_id):
        probabilities = self.stat_model.predict_proba(x[record_id])
        classes = self.stat_model.get_classes()

        class_prob_id = -1
        for c in range(len(classes)):
            if y[record_id] == classes[c]:
                class_prob_id = c
                break

        error = 1 - probabilities[class_prob_id]

        return error