import numpy as np
from sklearn.cluster import KMeans
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class ClusterTransformer(NumericTransformer):

    def __init__(self, column_id, number_clusters=10):
        NumericTransformer.__init__(self, column_id, "cluster", 1)
        self.number_clusters = number_clusters
        self.seed = 42

    def fit(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)
        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1
        newy = column_data.reshape(-1, 1)
        self.kmeans = KMeans(n_clusters=self.number_clusters, random_state=self.seed).fit(newy)

    def transform(self, dataset, ids):
        column_data = np.array(dataset.values[ids, self.column_id], dtype=np.float64)
        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1
        newy = column_data.reshape(-1, 1)
        return np.matrix(self.kmeans.predict(newy)).T

    def __str__(self):
        return self.__class__.__name__ + "_nr_clusters_" + str(self.number_clusters)
