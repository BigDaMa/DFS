from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.transformations.Transformation import Transformation
from typing import List
from fastfeature.transformations.UnaryTransformation import UnaryTransformation
from fastfeature.transformations.generators.NumpyClassGeneratorInvertible import NumpyClassGeneratorInvertible
from fastfeature.transformations.generators.NumpyClassGenerator import NumpyClassGenerator
from fastfeature.transformations.generators.HigherOrderCommutativeClassGenerator import HigherOrderCommutativeClassGenerator
from fastfeature.transformations.generators.NumpyBinaryClassGenerator import NumpyBinaryClassGenerator
from fastfeature.transformations.generators.GroupByThenGenerator import GroupByThenGenerator
from fastfeature.transformations.PandasDiscretizerTransformation import PandasDiscretizerTransformation
from fastfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
import numpy as np
from fastfeature.reader.Reader import Reader
from fastfeature.splitting.Splitter import Splitter
import time
from fastfeature.candidate_generation.explorekit.Generator import Generator
from fastfeature.candidates.RawFeature import RawFeature
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from fastfeature.plotting.plotter import cool_plotting


class SortExperiment:
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    #generate all possible combinations of features
    def generate(self):

        s = Splitter(train_fraction=[0.6, 10000000])

        self.dataset = Reader(self.dataset_config[0], self.dataset_config[1], s)
        raw_features = self.dataset.read()

        g = Generator(raw_features)
        self.candidates = g.generate_candidates()

    def my_arg_sort(self, seq):
        # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
        # non-lambda version by Tony Veijalainen
        return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]

    def stricly_increasing(self, a):
        print(np.all(np.diff((np.array(a), np.array(a))) >= 0))

    def run(self):
        self.generate()

        ids = self.my_arg_sort(self.candidates)

        depth = []
        transformations = []

        depth_max=-1
        t_max =-1
        for i in range(len(ids)):
            depth.append(self.candidates[ids[i]].get_transformation_depth())
            transformations.append(self.candidates[ids[i]].get_number_of_transformations())

            if transformations[-1] > t_max:
                t_max = transformations[-1]
            if transformations[-1] < t_max:
                raise ValueError('A very specific bad thing happened.')

            if depth[-1] > depth_max:
                depth_max = depth[-1]
            if depth[-1] < depth_max:
                raise ValueError('A very specific bad thing happened.')



        print(depth)
        print(transformations)

        print(self.stricly_increasing(depth))
        print(self.stricly_increasing(transformations))






if __name__ == '__main__':
    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
    selector = SortExperiment(dataset)
    selector.run()