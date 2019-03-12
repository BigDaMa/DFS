import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt

path = '/home/felix/phd/fastfeatures/results/11_03_incremental_construction'

cost_2_raw_features = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))

#find best candidate per cost

for c in cost_2_binary_transformed[3]:
    if str(c) == 'nanprod(nansum(exercise_induced_angina,number_of_major_vessels),thal)':
        print(str(c) + ": " + str(c.score))