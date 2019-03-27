import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt

#path = '/home/felix/phd/fastfeatures/results/11_03_incremental_construction'
#path = '/home/felix/phd/fastfeatures/results/12_03_incremental_03_threshold'
#path = '/home/felix/phd/fastfeatures/results/12_03_incremental_02_threshold'
path = '/tmp'
#path = '/home/felix/phd/fastfeatures/results/15_03_timed_transfusion'
#path = '/home/felix/phd/fastfeatures/results/15_03_timed_transfusion_node1'
#path = '/home/felix/phd/fastfeatures/results/16_03_test_transfusion_me'
#path = '/home/felix/phd/fastfeatures/results/18_03_banknote'
#path = '/home/felix/phd/fastfeatures/results/18_03_iris'
#path = '/home/felix/phd/fastfeatures/results/20_03_transfusion'


cost_2_raw_features: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))



def crawl_runtime(data: Dict[int, List[CandidateFeature]], name_to_runtime: Dict[str, float], complexity, runtime):
    for key, my_list in data.items():
        for c in my_list:
            if c.runtime_properties['score'] >= 0.0:
                name_to_runtime[str(c)] = c.runtime_properties['hyperparameters']['C']
                complexity.append(c.get_complexity())
                runtime.append(c.runtime_properties['hyperparameters']['C'])
    return name_to_runtime


name_to_runtime: Dict[str, float] = {}
complexity = []
runtime = []

crawl_runtime(cost_2_raw_features, name_to_runtime, complexity, runtime)
crawl_runtime(cost_2_unary_transformed, name_to_runtime, complexity, runtime)
crawl_runtime(cost_2_binary_transformed, name_to_runtime, complexity, runtime)
crawl_runtime(cost_2_combination, name_to_runtime, complexity, runtime)
crawl_runtime(cost_2_dropped_evaluated_candidates, name_to_runtime, complexity, runtime)

#print(name_to_runtime)

s = [(k, name_to_runtime[k]) for k in sorted(name_to_runtime, key=name_to_runtime.get, reverse=True)]

for k, v in s:
    print(str(k) + " -> " + str(v))

import matplotlib.pyplot as plt
plt.scatter(complexity, runtime)
plt.ylim((0,5))
plt.xlabel("Complexity")
plt.ylabel("Runtime")
plt.show()
