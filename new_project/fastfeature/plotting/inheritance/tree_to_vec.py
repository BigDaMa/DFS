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

c = all_candidates[1]
print(c)
print(all_data['new_scores'][names_all[c.get_name()]])



def candidate_feature_to_vec(candidate: CandidateFeature):
    return None








