import numpy as np
from sklearn.cluster import KMeans
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class ClusterDistTransformer(NumericTransformer):

    def __init__(self, column_id, number_clusters=10):
        NumericTransformer.__init__(self, column_id, "cluster_dist")
        self.number_clusters = number_clusters
        self.seed = 42

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1

        newy = column_data.reshape(-1, 1)
        self.kmeans = KMeans(n_clusters=self.number_clusters, random_state=self.seed).fit(newy)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1
        newy = column_data.reshape(-1, 1)
        return self.kmeans.transform(newy)
