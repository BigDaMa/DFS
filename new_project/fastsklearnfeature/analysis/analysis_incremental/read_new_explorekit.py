import pickle
from typing import List, Dict, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature

#path = '/home/felix/phd/fastfeatures/results/15_03_explorekit_transfusion'
#path = '/home/felix/phd/fastfeatures/results/15_03_1_node_explore'
path = '/home/felix/phd/fastfeatures/results/16_03_transfusion_explorekit'

candidates: Dict[int, CandidateFeature] = pickle.load(open(path + "/explorekit_results.p", "rb"))


for i in range(len(candidates)):
    print("i: " + str(i) + \
          " Candidate: " + str(candidates[i]) + \
          " test-score: " + str(candidates[i].runtime_properties['test_score']) + \
          " score: " + str(candidates[i].runtime_properties['score']) + \
          " time: " + str(candidates[i].runtime_properties['global_time']) + \
          " complexity: " + str(candidates[i].get_complexity())
          )

