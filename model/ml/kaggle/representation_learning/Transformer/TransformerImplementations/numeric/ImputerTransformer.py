import numpy as np
from sklearn.cluster import KMeans
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class InputerTransformer(NumericTransformer):

    def __init__(self, column_id):
        NumericTransformer.__init__(self, column_id, "imputer")
        self.number_clusters = 10
        self.seed = 42

    def fit(self, dataset, ids):
        '''
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        newy = column_data.reshape(-1, 1)
        self.kmeans = KMeans(n_clusters=self.number_clusters, random_state=self.seed).fit(newy)
        '''

    def transform(self, dataset, ids):
        '''
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        newy = column_data.reshape(-1, 1)
        return self.kmeans.predict(newy)
        '''
