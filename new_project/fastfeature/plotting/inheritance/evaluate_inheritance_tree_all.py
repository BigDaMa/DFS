import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np
from graphviz import Digraph
import tempfile
from fastfeature.plotting.inheritance.tree.MyNode import MyNode
import matplotlib.pyplot as plt


#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'

# hearts
#file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
#names_all_file = '/home/felix/phd/fastfeature_logs/newest_28_11/name2id.p'
#all_candidates_file = '/home/felix/phd/fastfeature_logs/newest_28_11/all_candidates.p'

# hearts - all features + fi
file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/chart.p'
names_all_file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/name2id.p'
all_candidates_file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/all_candiates.p'


# diabetes
#file = '/home/felix/phd/fastfeatures/results/diabetes/chart.p'
#names_all_file = '/home/felix/phd/fastfeatures/results/diabetes/name2id.p'
#all_candidates_file = '/home/felix/phd/fastfeatures/results/diabetes/all_candiates.p'




all_data = pickle.load(open(file, "rb"))
names_all: Dict[str, int] = pickle.load(open(names_all_file, "rb"))

candidate_id_to_stored_id= {}

for stored_id in range(len(all_data['ids'])):
    candidate_id_to_stored_id[all_data['ids'][stored_id]] = stored_id


all_candidates: List[CandidateFeature] = pickle.load(open(all_candidates_file, "rb"))


print(all_candidates[-1])
print(all_data['new_scores'][candidate_id_to_stored_id[names_all[all_candidates[-1].get_name()]]])





max_score = np.max(all_data['new_scores'])
min_score = np.min(all_data['new_scores'])

print('min score: ' + str(min_score))
print('starter: ' + str(all_data['start_score']))



start_score = all_data['start_score']
if start_score < 0:
    start_score = 0.7

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
    score = all_data['new_scores'][candidate_id_to_stored_id[names_all[all_candidates[i].get_name()]]]
    candidate_name_to_node[candidate.get_name()] = id

    size = -1.0
    if score > start_score:
        normed_value = (score - start_score) / (max_score - start_score)
        size = str(100 * normed_value)
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








