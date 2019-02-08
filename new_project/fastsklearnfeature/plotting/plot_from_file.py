import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import List, Dict, Any
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_sorted_by_complexity_fold20_hyper_opt_1045min.p'

#heart
#file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
#my_range = (0.72, 0.88)
# heart also raw features
file = '/home/felix/phd/fastfeatures/results/cluster_good_cv_fixed_group/all_data.p'
my_range = (0.50, 0.88)



#diabetes
#file = '/home/felix/phd/fastfeatures/results/diabetes/chart.p'
#my_range = (0.72, 0.78)

all_data = pickle.load(open(file, "rb"))


names = [str(r['candidate']) for r in all_data]
scores = [r['score'] for r in all_data]
runtimes = [r['time'] for r in all_data]
numbers_features_and_numbers_transformations = [(r['candidate'].get_number_of_transformations() + 1) for r in all_data]


cool_plotting(numbers_features_and_numbers_transformations,
              scores,
              names,
              0.0,
              my_range)

cool_plotting(runtimes,
              scores,
              names,
              0.0,
              my_range)