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



graph = nx.DiGraph()

max_score = max([v.runtime_properties['score'] for v in string2candidate.values()])
start_score = 0.5

def get_color_tuple(score, starter_score, max_score):
    Blues = plt.get_cmap('Blues')
    if score >= starter_score:
        normed_value = (score - starter_score) / (max_score - starter_score)
        color_tuple = Blues(normed_value)
        return color_tuple
    else:
        return (0, 0, 0)


graph.add_node('__root__')
for v in string2candidate.values():
	if not isinstance(v.transformation, IdentityTransformation):
		graph.add_node(str(v))
		graph.node[str(v)]['score'] = str(v.runtime_properties['score'])
		#graph.node[str(v)]['size'] = size
		graph.node[str(v)]['r'] = int(get_color_tuple(v.runtime_properties['score'], start_score, max_score)[0] * 256)
		graph.node[str(v)]['g'] = int(get_color_tuple(v.runtime_properties['score'], start_score, max_score)[1] * 256)
		graph.node[str(v)]['b'] = int(get_color_tuple(v.runtime_properties['score'], start_score, max_score)[2] * 256)


def generate_edges_for_candidate(graph: nx.DiGraph, candidate: CandidateFeature):
	if not isinstance(candidate, RawFeature):
		for p in candidate.parents:
			if not graph.has_edge(str(p), str(candidate)):
				#edge = graph.add_edge(str(p), str(candidate), weight=(candidate.get_complexity() - p.get_complexity()))
				edge = graph.add_edge(str(p), str(candidate))
			generate_edges_for_candidate(graph, p)
	else:
		edge = graph.add_edge('__root__', str(candidate))


for v in string2candidate.values():
	if not isinstance(v.transformation, IdentityTransformation):
		generate_edges_for_candidate(graph, v)


print(list(graph.successors('__root__')))
#print(list(graph.successors('thal')))

#show best 100 features

k=20

def prune_from_root(c: CandidateFeature):
	if isinstance(c, RawFeature):
		return True
	if not all([prune_from_root(p) for p in c.parents]):
		return False
	if c.runtime_properties['score'] > max([p.runtime_properties['score'] for p in c.parents]):
		return True
	else:
		return False

candidates = []
for c in string2candidate.values():
	if not isinstance(c, RawFeature) and prune_from_root(c):
		candidates.append(c)


complexities = np.array([c.runtime_properties['score'] for c in candidates]) * -1
sorted_ids = np.argsort(complexities)

if len(candidates) < k:
	k = len(candidates)

interesting_features = []

for my_i in range(k):
	interesting_features.append(candidates[sorted_ids[my_i]])
	print(str(candidates[sorted_ids[my_i]]) + " -> " + str(
		candidates[sorted_ids[my_i]].runtime_properties['score']) + " complexity: " + str(
		candidates[sorted_ids[my_i]].get_complexity()) + ' test_score: ' + str(
		candidates[sorted_ids[my_i]].runtime_properties['test_score']))




