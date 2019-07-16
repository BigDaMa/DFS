import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt
import numpy as np

path = '/home/felix/phd/fastfeatures/results/24_4_synthetic'


cost_2_raw_features = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))


def generate_str_candidate(candidates: Dict[int, List[CandidateFeature]], str2candidate: Dict[str,CandidateFeature]):
    for rep in candidates.values():
        for r in rep:
            str2candidate[str(r)] = r


str2candidate: Dict[str, CandidateFeature] = {}

generate_str_candidate(cost_2_raw_features, str2candidate)
print(str2candidate)
generate_str_candidate(cost_2_unary_transformed, str2candidate)
generate_str_candidate(cost_2_binary_transformed, str2candidate)
generate_str_candidate(cost_2_combination, str2candidate)


def find_and_print(my_str, str2candidate):
    try:
        print(my_str + ": " + str(str2candidate[my_str].runtime_properties['score']))
    except:
        print(my_str + ": not found")


#2 + 15*x1 + 3/(x2 - 1/x3)
find_and_print('x1', str2candidate)
find_and_print('x2', str2candidate)
find_and_print('x3', str2candidate)

find_and_print('1/(x3)', str2candidate)
find_and_print('-1*(x3)', str2candidate)
find_and_print('1/(-1*(x3))', str2candidate)
find_and_print('nansum(x2,1/(-1*(x3)))', str2candidate)
find_and_print('nansum(1/(-1*(x3)),x2)', str2candidate)



'''
#1/(x2 - 1/x3)
print('x2' + str(str2candidate["x2"].runtime_properties['score']))
print('x3' + str(str2candidate["x3"].runtime_properties['score']))
print('-1*(x3)' + str(str2candidate["-1*(x3)"].runtime_properties['score']))
print('1/(-1*(x3))' + str(str2candidate["1/(-1*(x3))"].runtime_properties['score']))
print('1/(x3)' + str(str2candidate["1/(-1*(x3))"].runtime_properties['score']))

print('nansum(x2,1/(-1*(x3)))' + str(str2candidate["nansum(x2,1/(-1*(x3)))"].runtime_properties['score']))
'''