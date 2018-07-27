import numpy as np

class RecordUncertainty(): #n_records_for_each_class

    def __init__(self, stat_model, number_classes=2):
        self.stat_model = stat_model
        self.number_classes = number_classes


    def get_reward(self, x, y, record_id):
        probabilities = self.stat_model.predict_proba(x[record_id])

        if self.number_classes > 2:
            topk = probabilities.argsort()[-2:][::-1]
            uncertainty = 1 - np.sum(probabilities[topk])
        else:
            topk = probabilities.argsort()[-1:][::-1]
            uncertainty = 1 - np.sum(probabilities[topk])

        return uncertainty