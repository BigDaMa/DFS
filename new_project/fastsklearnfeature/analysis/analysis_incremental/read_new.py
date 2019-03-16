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

cost_2_raw_features = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))

#find best candidate per cost

def get_max_candidate(candidates: Dict[int, List[CandidateFeature]], complexity, best_candidate: CandidateFeature):
    new_best_candidate = copy.deepcopy(best_candidate)
    if complexity in candidates:
        for candidate in candidates[complexity]:
            if candidate.runtime_properties['score'] > new_best_candidate.runtime_properties['score']:
                new_best_candidate = candidate
    return new_best_candidate


def count_smaller_or_equal(candidates: List[CandidateFeature], current_candidate: CandidateFeature):
    count_smaller_or_equal = 0
    for c in candidates:
        if c.runtime_properties['score'] <= current_candidate.runtime_properties['score']:
            count_smaller_or_equal += 1
    return count_smaller_or_equal


#P(Accuracy <= current) -> 1.0 = highest accuracy
def getAccuracyScore(current: CandidateFeature, complexity):
    count_smaller_or_equal_v = 0
    count_all = 0
    for c in range(1, complexity+1):
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_raw_features[c], current)
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_unary_transformed[c], current)
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_binary_transformed[c], current)
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_combination[c], current)

        count_all += len(cost_2_raw_features[c])
        count_all += len(cost_2_unary_transformed[c])
        count_all += len(cost_2_binary_transformed[c])
        count_all += len(cost_2_combination[c])


    return count_smaller_or_equal_v / float(count_all)

#P(Complexity >= current) -> 1.0 = lowest complexity
def getSimplicityScore(current: CandidateFeature, complexity):
    count_greater_or_equal_v = 0
    count_all = 0

    for c in range(1, complexity + 1):
        if c >= current.get_number_of_transformations():
            count_greater_or_equal_v += len(cost_2_raw_features[c])
            count_greater_or_equal_v += len(cost_2_unary_transformed[c])
            count_greater_or_equal_v += len(cost_2_binary_transformed[c])
            count_greater_or_equal_v += len(cost_2_combination[c])


        count_all += len(cost_2_raw_features[c])
        count_all += len(cost_2_unary_transformed[c])
        count_all += len(cost_2_binary_transformed[c])
        count_all += len(cost_2_combination[c])

    return count_greater_or_equal_v / float(count_all)


def plot_accuracy_histogram(complexity, color):
    all_candidates: List[CandidateFeature] = []
    for c in range(1, complexity + 1):
        all_candidates.extend(cost_2_raw_features[c])
        all_candidates.extend(cost_2_unary_transformed[c])
        all_candidates.extend(cost_2_binary_transformed[c])
        all_candidates.extend(cost_2_combination[c])

    scores = [c.runtime_properties['score'] for c in all_candidates]

    plt.hist(scores, bins=20, color=color)



def harmonic_mean(complexity, accuracy):
    return (2 * complexity * accuracy) / (complexity + accuracy)




raw_dropped = {}
unary_dropped = {}
binary_dropped = {}
combination_dropped = {}

#sort dropped by type:
for cost in range(1,8):
    unary_dropped[cost] = []
    combination_dropped[cost] = []
    raw_dropped[cost] = []
    binary_dropped[cost] = []

    if not cost in cost_2_dropped_evaluated_candidates:
        cost_2_dropped_evaluated_candidates[cost] = []

    for candidate in cost_2_dropped_evaluated_candidates[cost]:
        if isinstance(candidate, RawFeature):
            raw_dropped[cost].append(candidate)
        elif isinstance(candidate.transformation, UnaryTransformation):
            unary_dropped[cost].append(candidate)
        elif isinstance(candidate.transformation, IdentityTransformation):
            combination_dropped[cost].append(candidate)
        else:
            binary_dropped[cost].append(candidate)

    if not cost in cost_2_raw_features:
        cost_2_raw_features[cost] = []
    if not cost in cost_2_unary_transformed:
        cost_2_unary_transformed[cost] = []
    if not cost in cost_2_binary_transformed:
        cost_2_binary_transformed[cost] = []
    if not cost in cost_2_combination:
        cost_2_combination[cost] = []


    print("\nCost: " + str(cost))

    print("Evaluated Features: " + str(len(cost_2_raw_features[cost]) +
                                       len(cost_2_unary_transformed[cost]) +
                                       len(cost_2_binary_transformed[cost]) +
                                       len(cost_2_combination[cost]) +
                                       len(cost_2_dropped_evaluated_candidates[cost])) + " from that dropped " + str(len(cost_2_dropped_evaluated_candidates[cost])) + " -> remaining " + str(len(cost_2_raw_features[cost]) +
                                       len(cost_2_unary_transformed[cost]) +
                                       len(cost_2_binary_transformed[cost]) +
                                       len(cost_2_combination[cost])))

    print("From " + str(len(cost_2_raw_features[cost]) + len(raw_dropped[cost])) + " Raw Candidates " + str(len(raw_dropped[cost])) + " dropped")
    print("From " + str(len(cost_2_unary_transformed[cost]) + len(unary_dropped[cost])) + " Unary Candidates " + str(len(unary_dropped[cost])) + "  dropped")
    print("From " + str(len(cost_2_binary_transformed[cost]) + len(binary_dropped[cost])) + " Binary Candidates " + str(len(binary_dropped[cost])) + " dropped")
    print("From " + str(len(cost_2_combination[cost]) + len(combination_dropped[cost])) + " Combination Candidates " + str(len(combination_dropped[cost])) + " dropped")

print("\n")

best_pro_cost = {}

for c in range(1, 8):
    best_candidate = cost_2_raw_features[1][0]

    best_candidate = get_max_candidate(cost_2_raw_features, c, best_candidate)
    best_candidate = get_max_candidate(cost_2_unary_transformed, c, best_candidate)
    best_candidate = get_max_candidate(cost_2_binary_transformed, c, best_candidate)
    best_candidate = get_max_candidate(cost_2_combination, c, best_candidate)

    best_pro_cost[c] = copy.deepcopy(best_candidate)

    print(best_candidate.runtime_properties)
    print("complexity: " + str(c) + " " + \
          str(best_candidate) + \
          " cross-validation score: " + str(best_candidate.runtime_properties['score']) + \
          " test score: " + str(best_candidate.runtime_properties['test_score']) + \
          " layer time: " + str(best_candidate.runtime_properties['layer_end_time']) +\
          " real time: " + str(best_candidate.runtime_properties['global_time'])
          )

    acc_score = getAccuracyScore(best_pro_cost[c], c)
    simplicity_score = getSimplicityScore(best_pro_cost[c], c)

    print("acc: " + str(acc_score))
    print("simplicity: " + str(simplicity_score))
    print("Harmonic mean: " + str(harmonic_mean(simplicity_score, acc_score)))



for i in range(1, 8):
    acc_score = getAccuracyScore(best_pro_cost[i], 5)
    simplicity_score = getSimplicityScore(best_pro_cost[i], 5)

    print("\n\nstart: " + str(i))
    print("acc: " + str(acc_score))
    print("simplicity: " + str(simplicity_score))
    print("Harmonic mean: " + str(harmonic_mean(simplicity_score, acc_score)))

    print('just divid: ' + str(best_pro_cost[i].runtime_properties['score'] / i))

print("\n")

for c in cost_2_raw_features[1]:
    print(str(c) + ": " + str(c.runtime_properties['score']))

plot_accuracy_histogram(4, 'yellow')
plot_accuracy_histogram(3, 'green')
plot_accuracy_histogram(2, 'red')
plot_accuracy_histogram(1, 'blue')

plt.show()
