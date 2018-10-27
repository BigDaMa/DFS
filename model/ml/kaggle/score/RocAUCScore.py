from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

class RocAUCScore:
    def __init__(self, number_classes):
        self.number_classes = number_classes


    def score(self, y_true, y_pred):
        if self.number_classes > 2:
            return roc_auc_score(y_true, y_pred, average='micro')
        else:
            return roc_auc_score(y_true, y_pred)

    def get_scorer(self):
        if self.number_classes > 2:
            f_scorer = make_scorer(roc_auc_score, average='micro')
        else:
            f_scorer = make_scorer(roc_auc_score)
        return f_scorer