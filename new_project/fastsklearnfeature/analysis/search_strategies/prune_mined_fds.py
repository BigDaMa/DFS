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


#path = '/home/felix/phd/fastfeatures/results/heart_new_all'
#path = '/home/felix/phd/fastfeatures/results/heart_small'
#path = '/home/felix/phd/fastfeatures/results/transfusion'
path = '/home/felix/phd/fastfeatures/results/credit_4'
#path = '/tmp'



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
				string2candidate[str(v)] = v


string2candidate: Dict[str, CandidateFeature] = {}
extend_string2candidate(cost_2_raw_features, string2candidate, last_layer)
extend_string2candidate(cost_2_unary_transformed, string2candidate, last_layer)
extend_string2candidate(cost_2_binary_transformed, string2candidate, last_layer)
if load_combination:
	extend_string2candidate(cost_2_combination, string2candidate, last_layer)
extend_string2candidate(cost_2_dropped_evaluated_candidates, string2candidate, last_layer)

fds = open('/home/felix/Software/Metanome/deployment/target/test/results/HyFD-1.1-SNAPSHOT.jar2019-07-12T130622_fds', 'r')



id2name: Dict[str, str] = {}

target_id = -1

all_combinations = []

mode = 0
for line in fds:
	if '# COLUMN' in line:
		mode = 1
		continue
	if '# RESULTS' in line:
		mode = 2
		continue

	if mode == 1:
		tokens = line.strip().split('\t')
		id = tokens[-1]
		name = tokens[0][2:]
		id2name[id] = name
		if name == 'target':
			target_id = id

	if mode == 2:
		if '->' + target_id in line:
			id = line.find('->')
			all_fields = line[:id].split(',')

			feature_combination = []
			for feature_id in all_fields:
				feature_combination.append(string2candidate[id2name[feature_id]])
			all_combinations.append(feature_combination)


print(len(all_combinations))

def sum_complexity(combo: List[CandidateFeature]):
	my_sum = 0
	for c in combo:
		my_sum += c.get_complexity()
	return my_sum

def averageScore(combo: List[CandidateFeature]):
	my_sum = 0
	for c in combo:
		my_sum += c.runtime_properties['score']
	return my_sum / float(len(combo))

min_complexity = 100
min_complexity_combo = None

max_avg_score = -1
max_avg_score_combo = None

for l in all_combinations:
	if sum_complexity(l) < min_complexity:
		min_complexity = sum_complexity(l)
		min_complexity_combo = l

	if averageScore(l) > max_avg_score:
		max_avg_score = averageScore(l)
		max_avg_score_combo = l

print(min_complexity_combo)

print(max_avg_score_combo)

pickle.dump(max_avg_score_combo, open('/tmp/cover_features.p', 'wb+'))

pickle.dump(all_combinations, open('/tmp/all_fds.p', 'wb+'))