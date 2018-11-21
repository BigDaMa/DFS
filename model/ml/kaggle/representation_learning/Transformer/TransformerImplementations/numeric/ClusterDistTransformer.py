import numpy as np
from sklearn.cluster import KMeans
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.NumericTransformer import NumericTransformer


class ClusterDistTransformer(NumericTransformer):

    def __init__(self, column_id, number_clusters=10):
        NumericTransformer.__init__(self, column_id, "cluster_dist", number_clusters)
        self.number_clusters = number_clusters
        self.seed = 42

    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        newy = column_data.reshape(-1, 1)
        self.kmeans = KMeans(n_clusters=self.number_clusters, random_state=self.seed).fit(newy)

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id]).A1

        where_are_NaNs = np.isnan(column_data)
        column_data[where_are_NaNs] = -1

        newy = column_data.reshape(-1, 1)
        return self.kmeans.transform(newy)

    def __str__(self):
        return self.__class__.__name__ + "_nr_clusters_" + str(self.number_clusters)

    def get_feature_names(self, dataset):
        internal_names = []
        for cluster_i in range(self.number_clusters):
            internal_names.append(str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "clusterdistance_" + str(cluster_i))

        return internal_names
