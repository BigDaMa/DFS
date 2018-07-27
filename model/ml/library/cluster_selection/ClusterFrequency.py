import math
import numpy as np
import operator

class ClusterFrequency():
    def __init__(self, cluster_dict):
        self.id = 0
        size = {}

        for key, value in cluster_dict.iteritems():
            size[key] = len(value)

        self.ranked = sorted(size.items(), key=operator.itemgetter(1), reverse=True)


    def select_arm(self):
        cluster_id = self.ranked[self.id][0]
        self.id += 1

        if self.id >= len(self.ranked):
            self.id = 0

        return cluster_id


if __name__ == "__main__":

    clusters ={0: [1,2], 2: [3,4,5], 4:[1]}

    c = ClusterFrequency(clusters)