import pickle
from typing import Dict, List
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
import numpy as np
import matplotlib.pyplot as plt

file = '/home/felix/phd/fastfeatures/results/cluster_good_cv_fixed_group/all_data.p'
all_data = pickle.load(open(file, "rb"))

all_candidates: List[CandidateFeature] = [r['candidate'] for r in all_data]
scores = [r['score'] for r in all_data]



max_score = np.max(scores)
min_score = np.min(scores)

start_score = 0.75

import networkx as nx
graph = nx.Graph()


def get_color_tuple(score, starter_score, max_score):
    Blues = plt.get_cmap('Blues')
    if score >= starter_score:
        normed_value = (score - starter_score) / (max_score - starter_score)
        color_tuple = Blues(normed_value)
        return color_tuple
    else:
        return (0, 0, 0)

candidate_name_to_node = {}
for i in range(len(all_candidates)):
    id = str(i)
    candidate = all_candidates[i]
    score = scores[i]
    candidate_name_to_node[candidate.get_name()] = id

    size = -1.0
    if score > start_score:
        normed_value = (score - start_score) / (max_score - start_score)
        size = str(np.square(100 * normed_value))
    else:
        size = str(1)

    graph.add_node(id)
    graph.node[id]['label'] = candidate.get_name()
    graph.node[id]['score'] = str(score)
    graph.node[id]['size'] = size
    graph.node[id]['r'] = int(get_color_tuple(score, start_score, max_score)[0] * 256)
    graph.node[id]['g'] = int(get_color_tuple(score, start_score, max_score)[1] * 256)
    graph.node[id]['b'] = int(get_color_tuple(score, start_score, max_score)[2] * 256)


def generate_edges_for_candidate(graph, candidate):
    id = candidate_name_to_node[candidate.get_name()]
    if not isinstance(candidate, RawFeature):
        for p in range(len(candidate.parents)):
            if candidate.parents[p].get_name() in candidate_name_to_node:
                if not graph.has_edge(str(id), candidate_name_to_node[candidate.parents[p].get_name()]):
                    edge = graph.add_edge(str(id), candidate_name_to_node[candidate.parents[p].get_name()])
                generate_edges_for_candidate(graph, candidate.parents[p])


for i in range(len(all_candidates)):
    generate_edges_for_candidate(graph, all_candidates[i])



print('----')
print(nx.edges(graph, ['35596']))


for isolate in nx.isolates(graph):
    print(isolate)

print("number isolates: " + str(len(list(nx.isolates(graph)))))

nx.write_graphml(graph, '/tmp/node_new.graphml')








