import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored




path = '/home/felix/phd/fastfeatures/results/29_4_transfusion'


cost_2_raw_features = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))

#check how many representations violate monotonicity and which transformations are more likely to cause these violations


def get_reps_of_complexity(c: int) -> List[CandidateFeature]:
    my_list = []
    if c in cost_2_raw_features:
        my_list.extend(cost_2_raw_features[c])
    if c in cost_2_unary_transformed:
        my_list.extend(cost_2_unary_transformed[c])
    if c in cost_2_binary_transformed:
        my_list.extend(cost_2_binary_transformed[c])
    if c in cost_2_combination:
        my_list.extend(cost_2_combination[c])
    return my_list

def is_monotone(candidate: CandidateFeature):
    if isinstance(candidate, RawFeature):
        return True
    max_p = max([p.runtime_properties['score'] for p in candidate.parents])
    if candidate.runtime_properties['score'] < max_p:
        return False
    return all([is_monotone(p) for p in candidate.parents])

def is_strict_monotone(candidate: CandidateFeature):
    if isinstance(candidate, RawFeature):
        return True
    max_p = max([p.runtime_properties['score'] for p in candidate.parents])
    if candidate.runtime_properties['score'] <= max_p:
        return False
    return all([is_monotone(p) for p in candidate.parents])

def get_score_tree(candidate: CandidateFeature, depth = 0):
    print(''.join(['\t' for i in range(depth)]) + str(candidate) + ": " + str(candidate.runtime_properties['score']))
    for p in candidate.parents:
        get_score_tree(p, depth+1)


#get all leaf representations
leaf_reps: Dict[str, CandidateFeature] = {}
for c in range(1,10):
    reps_c = get_reps_of_complexity(c)
    for rep in reps_c:
        for p in rep.parents:
            if str(p) in leaf_reps:
                del leaf_reps[str(p)]
        leaf_reps[str(rep)] = rep

#print(leaf_reps)

#get one score sequence per leaf_rep

monotonicity_violations = 0
for rep in leaf_reps.values():
    if not is_monotone(rep):
        #print(rep)
        monotonicity_violations += 1

print("violation fraction: " + str(monotonicity_violations / float(len(leaf_reps))))


#rank representations by score and check monotonicity
all_reps = []
for c in range(1, 10):
    all_reps.extend(get_reps_of_complexity(c))

scores = np.array([rep.runtime_properties['score'] for rep in all_reps])*-1
ids = np.argsort(scores)

for i in range(100):
    current_rep = all_reps[ids[i]]

    if not is_monotone(current_rep):
        print(colored('number: ' + str(i) + ': ' + str(current_rep) + ' ' + str(current_rep.runtime_properties['score']), 'red'))

        get_score_tree(current_rep)

    else:
        print(
            colored('number: ' + str(i) + ': ' + str(current_rep) + ' ' + str(current_rep.runtime_properties['score']),
                    'green'))


cumulative_distribution = []
for i in range(len(all_reps)):
    current_rep = all_reps[ids[i]]
    is_mono = int(is_monotone(current_rep))

    if i == 0:
        cumulative_distribution.append(0 + is_mono)
    else:
        cumulative_distribution.append(cumulative_distribution[-1] + is_mono)


hist_data_fail = []
hist_data = []
for i in range(len(all_reps)):
    current_rep = all_reps[ids[i]]
    if not is_strict_monotone(current_rep):
        hist_data_fail.append(all_reps[ids[i]].runtime_properties['score'])
    else:
        hist_data.append(all_reps[ids[i]].runtime_properties['score'])


#plt.plot(range(len(all_reps)), cumulative_distribution)
#plt.plot(scores[ids]*-1, np.array(cumulative_distribution) / float(len(cumulative_distribution)))

plt.hist(hist_data_fail, bins=50, color='red')
plt.hist(hist_data, bins=50, color='blue')
plt.show()
