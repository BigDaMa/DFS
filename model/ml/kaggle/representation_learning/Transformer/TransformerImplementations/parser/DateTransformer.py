import numpy as np
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AllTransformer import AllTransformer
import dateutil.parser
import time

class DateTransformer(AllTransformer):

    def __init__(self, column_id):
        AllTransformer.__init__(self, column_id, "timestamp")

    def transform(self, dataset, ids):
        try:
            column_data = dataset[dataset.columns[self.column_id]].copy()
            as_timestamp = column_data.apply(lambda x: time.mktime((dateutil.parser.parse(x)).timetuple())).values[ids]

            return np.matrix(as_timestamp).T
        except ValueError:
            self.applicable = False
            return None
        except TypeError:
            self.applicable = False
            return None