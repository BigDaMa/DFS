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
path = '/home/felix/phd/fastfeatures/results/credit'



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

#print('(nanstd(num_dependents) GroupyBy purpose)')
#print(string2candidate['(nanstd(num_dependents) GroupyBy purpose)'].runtime_properties['score'])


#find features that destroy monotonicity
count_violations = 0

count_violations_per_transformation = {}

for c in string2candidate.values():
	if c.get_complexity() >= 3:
		#print('' + str(c))
		parents = list(c.parents)
		#print('\t' + str(parents))
		id = np.argmax(np.array([p.runtime_properties['score']for p in parents]))
		if parents[id].runtime_properties['score'] < c.runtime_properties['score']:
			parent_parents = list(parents[id].parents)
			if len(parent_parents) >= 1:
				#print('\t\t' + str(parent_parents))
				new_id = np.argmax(np.array([p.runtime_properties['score'] for p in parent_parents]))
				if parent_parents[new_id].runtime_properties['score'] < c.runtime_properties['score'] and parent_parents[new_id].runtime_properties['score'] > parents[id].runtime_properties['score']:
					print(c)
					print(str(c) + ': ' + str(c.runtime_properties['score']))
					print('\t' + str(parents[id]) + ': ' + str(parents[id].runtime_properties['score']))
					print('\t\t' + str(parent_parents[new_id]) + ': ' + str(parent_parents[new_id].runtime_properties['score']))
					print('')
					count_violations +=1
					if not c.transformation.name in count_violations_per_transformation:
						count_violations_per_transformation[c.transformation.name] = 0
					count_violations_per_transformation[c.transformation.name] += 1

print(count_violations_per_transformation)

print("violations: " + str(count_violations) + " of " + str(len(string2candidate)) + ' representations')





