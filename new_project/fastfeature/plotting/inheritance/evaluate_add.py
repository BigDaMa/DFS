import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np
from graphviz import Digraph
import tempfile
from fastfeature.plotting.inheritance.tree.MyNode import Node


#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
names_all_file = '/home/felix/phd/fastfeature_logs/newest_28_11/name2id.p'



all_data = pickle.load(open(file, "rb"))
names_all: Dict[str, int]  = pickle.load(open(names_all_file, "rb"))




all_candidates: List[CandidateFeature] = pickle.load(open('/home/felix/phd/fastfeature_logs/newest_28_11/all_candidates.p', "rb"))


print(all_candidates[-1])
print(all_data['new_scores'][names_all[all_candidates[-1].get_name()]])


#tree
root = Node(-1, None, -1)

candidate_name_to_node: Dict[str, Node] = {}

counter_tree_node_id = 0
for i in range(len(all_candidates)):

    if all_data['new_scores'][names_all[all_candidates[i].get_name()]] > 0:
        current_node = Node(i, all_candidates[i], all_data['new_scores'][names_all[all_candidates[i].get_name()]])
        candidate_name_to_node[all_candidates[i].get_name()] = current_node

        parent_found = False
        for p in range(len(all_candidates[i].parents)):
            if all_candidates[i].parents[p].get_name() in candidate_name_to_node:
                candidate_name_to_node[all_candidates[i].parents[p].get_name()].add_child(current_node)
                parent_found = True
        if parent_found == False:
            root.add_child(current_node)





def return_all_input_output_accuracy_for_add(node: Node, input_output_combination=[]):
    if node.candidate != None:
        #print(node.candidate.transformation.name)
        #if node.candidate.transformation.name == 'nansum':
        #if node.candidate.transformation.name == 'nanprod':
        #if node.candidate.transformation.name == 'divide':
        #if node.candidate.transformation.name == 'subtract':
        #if node.candidate.transformation.name == 'MinMaxScaling':
        #if node.candidate.transformation.name == 'Discretizer':
        if node.candidate.transformation.name == 'GroupByThenlen':
            try:
                key = tuple(candidate_name_to_node[p.get_name()].score for p in node.candidate.parents)
                input_output_combination.append((key, node.score))
            except:
                pass
    for c in node.children:
        return_all_input_output_accuracy_for_add(c, input_output_combination)
    return input_output_combination

input_output_combination = return_all_input_output_accuracy_for_add(root)


print(input_output_combination)

print("number of combinations: " + str(len(input_output_combination)))



input_product = list(np.prod(np.array(list(k[0]))) for k in input_output_combination)
#input_product = list(np.sum(np.array(list(k[0]))) for k in input_output_combination)
#input_product = list(np.average(np.array(list(k[0]))) for k in input_output_combination)
#input_product = list(np.var(np.array(list(k[0]))) for k in input_output_combination)
output = list(k[1] for k in input_output_combination)


import matplotlib.pyplot as plt

print(input_product)

plt.plot(input_product, output, 'ro')
plt.axhline(y=all_data['start_score'])

plt.xlabel('aggregating input scores')
plt.ylabel('output')
plt.title('About as simple as it gets, folks')
#plt.grid(True)
plt.show()











