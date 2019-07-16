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


#path = '/home/felix/phd/fastfeatures/results/heart_new_all'
#path = '/home/felix/phd/fastfeatures/results/heart_small'
#path = '/home/felix/phd/fastfeatures/results/transfusion'
path = '/home/felix/phd/fastfeatures/results/credit_4'
#path = '/tmp'

target_test_folds_global = pickle.load(open('/tmp/test_groundtruth.p', 'rb'))



load_combination = True


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



def is_consistent(candidate: CandidateFeature, target_test_folds_global):
	#which keys are inconsistent

	key2labels = {}

	for fold in range(len(candidate.runtime_properties['test_transformed'])):
		for row_id in range(len(candidate.runtime_properties['test_transformed'][fold])):
			key = tuple(candidate.runtime_properties['test_transformed'][fold][row_id])
			#print(key)

			if not key in key2labels:
				key2labels[key] = set()
			key2labels[key].add(target_test_folds_global[fold][row_id])

	inconsistent_values = set()
	for k, v in key2labels.items():
		if len(v) > 1:
			inconsistent_values.add(k)

	consistent_values = []
	for fold in range(len(candidate.runtime_properties['test_transformed'])):
		for row_id in range(len(candidate.runtime_properties['test_transformed'][fold])):
			key = tuple(candidate.runtime_properties['test_transformed'][fold][row_id])

			consistent_values.append(not key in inconsistent_values)

	return consistent_values



#consistent_values = is_consistent(string2candidate['{checking_status, duration}'], target_test_folds_global)

#print(consistent_values)















predictions2names = {}

counter = 0

for c in string2candidate.values():
	if 'test_fold_predictions' in c.runtime_properties:

		is_correct_list = []
		for fold in range(10):
			is_correct_list.extend(c.runtime_properties['test_fold_predictions'][fold].flatten())

		is_consistent_list = is_consistent(c, target_test_folds_global)

		both = np.logical_and(is_correct_list, is_consistent_list)
		#print(both)

		materialized = tuple(both)

		my_list = []
		for i in range(len(materialized)):
			if materialized[i] == True:
				my_list.append(i)

		my_set = frozenset(my_list)
		predictions2names[my_set] = c

#http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
def set_cover(universe, subsets):
	"""Find a family of subsets that covers the universal set"""
	elements = set(e for s in subsets for e in s)
	# Check the subsets cover the universe
	if elements != universe:
		return None
	covered = set()
	cover = []
	# Greedily add the subsets with the most uncovered points
	while covered != elements:
		subset = max(subsets, key=lambda s: len(s - covered))
		cover.append(subset)
		covered |= subset

	return cover



universe = set(range(len(materialized)))
subsets = list(predictions2names.keys())
cover = set_cover(universe, subsets)

feature_list: List[CandidateFeature] = []
complexity = 0
for cover_set in cover:
	complexity += predictions2names[cover_set].get_complexity()
	print(str(predictions2names[cover_set]) + ' score: ' + str(predictions2names[cover_set].runtime_properties['score']))
	feature_list.append(predictions2names[cover_set])
print("total complexity: " + str(complexity))


#w = [predictions2names[s].get_complexity() for s in subsets]
w = [1/predictions2names[s].runtime_properties['score'] for s in subsets]

pickle.dump(feature_list, open('/tmp/cover_features1.p', 'wb+'))


selected, cost = weightedsetcover(subsets, w)

print(selected)

feature_list: List[CandidateFeature] = []
for cover_set in selected:
	print(str(predictions2names[subsets[cover_set]]) + ' score: ' + str(predictions2names[subsets[cover_set]].runtime_properties['score']))
	feature_list.append(predictions2names[subsets[cover_set]])


pickle.dump(feature_list, open('/tmp/cover_features.p', 'wb+'))

