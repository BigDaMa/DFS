import pickle
from fastfeature.plotting.plotter import cool_plotting
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from typing import List, Dict, Any
import numpy as np

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_sorted_by_complexity_fold20_hyper_opt_1045min.p'

#heart
#file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
#my_range = (0.72, 0.88)
# heart also raw features
file = '/home/felix/phd/fastfeatures/results/cluster_good_cv/all_data.p'
my_range = (0.50, 0.88)



#diabetes
#file = '/home/felix/phd/fastfeatures/results/diabetes/chart.p'
#my_range = (0.72, 0.78)

all_data = pickle.load(open(file, "rb"))


names = [str(r['candidate']) for r in all_data]
scores = [r['score'] for r in all_data]
runtimes = [r['time'] for r in all_data]
candidates: List[CandidateFeature] = [r['candidate'] for r in all_data]

group_by_transformation = {}
for i in range(len(candidates)):
    c = candidates[i]
    if not isinstance(c, RawFeature):
        if not c.transformation.name in group_by_transformation:
            group_by_transformation[c.transformation.name] = []
        group_by_transformation[c.transformation.name].append(scores[i])


keys = []
avg_score = []
for k,v in group_by_transformation.items():
    keys.append(k)
    avg_score.append(np.mean(v))
    print(k + ': mean ' + str(np.mean(v)) + " max " + str(np.max(v)))