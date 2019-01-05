import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np
from graphviz import Digraph
import tempfile
from fastfeature.plotting.inheritance.tree.MyNode import MyNode


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




all_candidates: List[CandidateFeature] = pickle.load(open(all_candidates_file, "rb"))


print(all_candidates[-1])
print(all_data['new_scores'][names_all[all_candidates[-1].get_name()]])


#tree
root = MyNode(-1, None, -1)

candidate_name_to_node: Dict[str, MyNode] = {}

counter_tree_node_id = 0
for i in range(len(all_candidates)):

    #if all_data['new_scores'][names_all[all_candidates[i].get_name()]] > 0:
    current_node = MyNode(i, all_candidates[i], all_data['new_scores'][names_all[all_candidates[i].get_name()]])
    candidate_name_to_node[all_candidates[i].get_name()] = current_node

    if isinstance(all_candidates[i], RawFeature):
        root.add_child(current_node)
    else:
        parent_found = False
        for p in range(len(all_candidates[i].parents)):
            if all_candidates[i].parents[p].get_name() in candidate_name_to_node:
                candidate_name_to_node[all_candidates[i].parents[p].get_name()].add_child(current_node)
                parent_found = True
            else:
                if not isinstance(all_candidates[i].parents[p], RawFeature):
                    print("failed: " + str(all_candidates[i].parents[p].get_name()) + " - " + str(all_data['new_scores'][names_all[all_candidates[i].parents[p].get_name()]]))
        if parent_found == False:
            root.add_child(current_node)


max_score = np.max(all_data['new_scores'])
min_score = np.min(all_data['new_scores'])

print('min score: ' + str(min_score))
print('starter: ' + str(all_data['start_score']))



starter = all_data['start_score']

if starter < 0:
    starter = 0.5

#graph = root.create_graph(start_score=all_data['start_score'], max_score=max_score)
#graph.render(tempfile.mktemp('.gv'), view=False)

#graph = root.create_graph(start_score=all_data['start_score'], max_score=max_score)

import networkx as nx
G = nx.Graph()
graph = root.create_graph_graphml(G, start_score=starter, max_score=max_score)

for isolate in nx.isolates(graph):
    print(isolate)

print("number isolates: " + str(len(list(nx.isolates(graph)))))

nx.write_graphml(graph, '/tmp/node_size_1.graphml')







