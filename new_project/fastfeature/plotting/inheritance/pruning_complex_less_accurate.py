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
import numpy as np
from fastfeature.plotting.plotter import pruned_plotting

from fastfeature.plotting.inheritance.candidates_to_graph import candidates_to_graph
from fastfeature.plotting.inheritance.filters.check_predecessors_success import check_if_predecessors_were_more_successful
from fastfeature.plotting.inheritance.filters.enrich_candidates import enrich_candidates_all
from fastfeature.plotting.inheritance.filters.prune_by_raw_feature_combinations_and_interpretability import filter_best_combinations
from fastfeature.plotting.inheritance.filters.filter_best_by_transformation import filter_best_accuracy_per_transformation


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


def depth_greater2(c: CandidateFeature):
    return c.get_transformation_depth() > 2

def do_not_use_feature(c: CandidateFeature):
    return 'resting_electrocardiographic_results' in c.get_name()



candidate_id_to_stored_id= {}
for stored_id in range(len(all_data['ids'])):
    candidate_id_to_stored_id[all_data['ids'][stored_id]] = stored_id



all_pruned_candidates = []


#hierarchy based pruning
not_pruned_candidates, pruned_candidates = check_if_predecessors_were_more_successful(all_candidates, all_data, candidate_id_to_stored_id, names_all)
all_pruned_candidates.extend(pruned_candidates)

#raw feature combination based search
not_pruned_candidates, pruned_candidates = filter_best_combinations(not_pruned_candidates, all_data, candidate_id_to_stored_id, names_all)
all_pruned_candidates.extend(pruned_candidates)


names_not_pruned, scores_not_pruned, interpretabilities_not_pruned, names_pruned, scores_pruned, interpretabilities_pruned = enrich_candidates_all(not_pruned_candidates, all_pruned_candidates, all_data, candidate_id_to_stored_id, names_all)


number_transaction_max_accurate_candidate, number_transaction_max_accurate_score = filter_best_accuracy_per_transformation(not_pruned_candidates,
                                            all_data,
                                            candidate_id_to_stored_id,
                                            names_all)

print(number_transaction_max_accurate_candidate)
print(number_transaction_max_accurate_score)

candidates_to_graph(not_pruned_candidates, scores_not_pruned, 0.7, '/tmp/graph_final.graphml', connected=True)
#candidates_to_graph(not_pruned_candidates, scores_not_pruned, 0.7, '/tmp/graph_final.graphml', connected=False)


print("Pruned: " + str(len(names_pruned)))
print("Not Pruned: "+ str(len(names_not_pruned)))




pruned_plotting(interpretabilities_pruned,
              scores_pruned,
              names_pruned,
              interpretabilities_not_pruned,
              scores_not_pruned,
              names_not_pruned,
              0.0,
              [0.0, 1.0])











