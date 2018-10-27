from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

class F1Score:
    def __init__(self, number_classes):
        self.number_classes = number_classes


    def score(self, y_true, y_pred):
        if self.number_classes > 2:
            return f1_score(y_true, y_pred, average='micro')
        else:
            return f1_score(y_true, y_pred)

    def get_scorer(self):
        if self.number_classes > 2:
            f_scorer = make_scorer(f1_score, average='micro')
        else:
            f_scorer = make_scorer(f1_score)
        return f_scorer