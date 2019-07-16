import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fastsklearnfeature.analysis.search_strategies.weighted_set_cover.weightedsetcover import weightedsetcover
from fastsklearnfeature.feature_selection.openml_wrapper.openMLdict import openMLname2task
from fastsklearnfeature.reader.OnlineOpenMLReader import OnlineOpenMLReader
import pandas as pd


#path = '/home/felix/phd/fastfeatures/results/heart_new_all'
#path = '/home/felix/phd/fastfeatures/results/heart_small'
#path = '/home/felix/phd/fastfeatures/results/transfusion'
path = '/home/felix/phd/fastfeatures/results/credit_4'
#path = '/tmp'

target_test_folds_global = pickle.load(open('/tmp/test_groundtruth.p', 'rb'))



load_combination = False


cost_2_raw_features: Dict[int, List[RawFeature]] = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_binary.p", "rb"))
if load_combination:
	cost_2_combination: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))

#build tree from logged data

#get last layer:
all_layers = list(cost_2_raw_features.keys())
all_layers.extend(list(cost_2_unary_transformed.keys()))
all_layers.extend(list(cost_2_binary_transformed.keys()))
if load_combination:
	all_layers.extend(list(cost_2_combination.keys()))
all_layers.extend(list(cost_2_dropped_evaluated_candidates.keys()))

last_layer = max(all_layers)
print('last layer: ' + str(last_layer))

#create string2candidate dictionary
def extend_string2candidate(my_dict: Dict[int, List[CandidateFeature]], string2candidate: Dict[str, CandidateFeature], last_layer):
	for c in range(0, last_layer + 1):
		if c in my_dict:
			for v in my_dict[c]:
				if v.get_complexity() < 5:
					string2candidate[str(v)] = v


string2candidate: Dict[str, CandidateFeature] = {}
extend_string2candidate(cost_2_raw_features, string2candidate, last_layer)
extend_string2candidate(cost_2_unary_transformed, string2candidate, last_layer)
extend_string2candidate(cost_2_binary_transformed, string2candidate, last_layer)
if load_combination:
	extend_string2candidate(cost_2_combination, string2candidate, last_layer)
extend_string2candidate(cost_2_dropped_evaluated_candidates, string2candidate, last_layer)




all_features = []
names = []

score_list = []

for name, c in string2candidate.items():
	unique_values = 0
	if 'test_transformed' in c.runtime_properties:
		materialized_all = []
		for fold in range(10):
			materialized_all.extend(c.runtime_properties['test_transformed'][fold].flatten())
		all_features.append(materialized_all)
		names.append(name)
		score_list.append(c.runtime_properties['score'])

N = 60

ids = np.argsort(np.array(score_list)*-1)
subset_features = [all_features[i] for i in ids[0:N]]
subset_names = [names[i] for i in ids[0:N]]



materialized_all = []
for fold in range(10):
	materialized_all.extend(target_test_folds_global[fold].flatten())
subset_features.append(materialized_all)
subset_names.append('target')


all = np.matrix(subset_features).transpose()

df = pd.DataFrame(data=all, columns=subset_names)

import csv
df.to_csv('/home/felix/Software/Metanome/deployment/target/test/backend/WEB-INF/classes/inputData/feature/credit_features.csv', index=False, header=True, quoting=csv.QUOTE_ALL)

