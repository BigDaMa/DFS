import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt
import numpy as np

#path = '/tmp'
#path = '/home/felix/phd/fastfeatures/results/1_4_german_credit'
path = '/home/felix/phd/fastfeatures/results/2_4_transfusion_with_predictions'


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


def count_smaller_or_equal(candidates: List[CandidateFeature], current_score):
    count_smaller_or_equal = 0
    for c in candidates:
        if c.runtime_properties['score'] <= current_score:
            count_smaller_or_equal += 1
    return count_smaller_or_equal


def extract_accuracy(my_list: Dict[int, List[CandidateFeature]]):
    accuracy_list: List[float] = []
    for k,v in my_list.items():
        accuracy_list.extend([rep.runtime_properties['score'] for rep in v])
    return accuracy_list

def get_all_accuracy():
    accuracy_list: List[float] = []

    accuracy_list.extend(extract_accuracy(cost_2_raw_features))
    accuracy_list.extend(extract_accuracy(cost_2_unary_transformed))
    accuracy_list.extend(extract_accuracy(cost_2_binary_transformed))
    accuracy_list.extend(extract_accuracy(cost_2_combination))
    return accuracy_list

def extract_complexity(my_list: Dict[int, List[CandidateFeature]]):
    accuracy_list: List[float] = []
    for k,v in my_list.items():
        accuracy_list.extend([rep.get_complexity() for rep in v])
    return accuracy_list

def get_all_complexity():
    accuracy_list: List[float] = []

    accuracy_list.extend(extract_complexity(cost_2_raw_features))
    accuracy_list.extend(extract_complexity(cost_2_unary_transformed))
    accuracy_list.extend(extract_complexity(cost_2_binary_transformed))
    accuracy_list.extend(extract_complexity(cost_2_combination))
    return accuracy_list


#P(Accuracy <= current) -> 1.0 = highest accuracy
def getAccuracyScore(current_score, complexity):
    count_smaller_or_equal_v = 0
    count_all = 0
    for c in range(1, complexity+1):
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_raw_features[c], current_score)
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_unary_transformed[c], current_score)
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_binary_transformed[c], current_score)
        count_smaller_or_equal_v += count_smaller_or_equal(cost_2_combination[c], current_score)

        count_all += len(cost_2_raw_features[c])
        count_all += len(cost_2_unary_transformed[c])
        count_all += len(cost_2_binary_transformed[c])
        count_all += len(cost_2_combination[c])


    return count_smaller_or_equal_v / float(count_all)

#P(Complexity >= current) -> 1.0 = lowest complexity
def getSimplicityScore(current_complexity, complexity):
    count_greater_or_equal_v = 0
    count_all = 0

    for c in range(1, complexity + 1):
        if c >= current_complexity:
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
cost = 1
while True:
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
    cost += 1

    if not cost in cost_2_raw_features and \
       not cost in cost_2_unary_transformed and \
       not cost in cost_2_binary_transformed and \
       not cost in cost_2_combination:
        break

print("\n")

best_pro_cost = {}

best_pro_cost_real = {}

c = 1
while True:
    best_candidate = cost_2_raw_features[1][0]

    best_candidate = get_max_candidate(cost_2_raw_features, c, best_candidate)
    best_candidate = get_max_candidate(cost_2_unary_transformed, c, best_candidate)
    best_candidate = get_max_candidate(cost_2_binary_transformed, c, best_candidate)
    best_candidate = get_max_candidate(cost_2_combination, c, best_candidate)

    best_pro_cost[c] = copy.deepcopy(best_candidate)
    if (c-1) in best_pro_cost_real and best_pro_cost_real[c-1].runtime_properties['score'] >= best_pro_cost[c].runtime_properties['score']:
        best_pro_cost_real[c] = copy.deepcopy(best_pro_cost_real[c-1])
    else:
        best_pro_cost_real[c] = best_pro_cost[c]

    best_candidate = best_pro_cost_real[c]


    #print(best_candidate.runtime_properties)
    print("\ncomplexity: " + str(c) + " " + \
          str(best_candidate) + \
          " cross-validation score: " + str(best_candidate.runtime_properties['score']) + \
          " test score: " + str(best_candidate.runtime_properties['test_score']) + \
          " layer time: " + str(best_candidate.runtime_properties['layer_end_time']) +\
          " real time: " + str(best_candidate.runtime_properties['global_time'])
          )

    '''
    acc_score = getAccuracyScore(best_pro_cost[c], c)
    simplicity_score = getSimplicityScore(best_pro_cost[c], c)

    print("acc: " + str(acc_score))
    print("simplicity: " + str(simplicity_score))
    print("Harmonic mean: " + str(harmonic_mean(simplicity_score, acc_score)))
    '''

    c += 1
    if not c in cost_2_raw_features and \
       not c in cost_2_unary_transformed and \
       not c in cost_2_binary_transformed and \
       not c in cost_2_combination:
        break


print("\n\n\n")

c = 1
while True:
    best_harmonic_mean = -1
    best_harmonic_rep = None

    print("Complexity: " + str(c))
    print("------------------------------------------------------")
    for i in range(1, c+1):
        acc_score = getAccuracyScore(best_pro_cost_real[i].runtime_properties['score'], c)
        #acc_score = best_pro_cost_real[i].runtime_properties['score']
        simplicity_score = getSimplicityScore(best_pro_cost_real[i].get_complexity(), c)

        h = harmonic_mean(simplicity_score, acc_score)
        if h > best_harmonic_mean:
            best_harmonic_mean = h
            best_harmonic_rep = best_pro_cost_real[i]

        print('candidate:' + str(best_pro_cost_real[i]))
        print("acc: " + str(acc_score))
        print("simplicity: " + str(simplicity_score))
        print("harmonic mean: " + str(h))
        print("\n")

    print("\n\n\n")

    c += 1
    if not c in cost_2_raw_features and \
            not c in cost_2_unary_transformed and \
            not c in cost_2_binary_transformed and \
            not c in cost_2_combination:
        break


print("Best harmonic representation: " + str(best_harmonic_rep))

# plot distributions
last_c = c -1

acc_range = np.arange(0.01, 1.0, 0.01)
acc_score = [getAccuracyScore(score, last_c) for score in acc_range]


plt.suptitle('Harmonic mean: ' + str(best_harmonic_rep))
plt.subplot(2, 2, 1)
plt.plot(acc_range, acc_score)
plt.axvline(x=best_harmonic_rep.runtime_properties['score'], color='red')
plt.xlabel('Accuracy(F1)')
plt.ylabel('Accuracy Score: P(Accuracy <= x)')
plt.xlim((0.0, 1.0))

complexity_range = np.arange(1, c+1, 1)
simplicity_score = [getSimplicityScore(complexity, last_c) for complexity in complexity_range]
plt.subplot(2, 2, 2)
plt.plot(complexity_range*-1, simplicity_score)
plt.axvline(x=best_harmonic_rep.get_complexity()*-1, color='red')
plt.xlabel('-1 * Complexity')
plt.ylabel('Simplicity Score: P(Complexity >= x)')
plt.xlim((-6, -1))


plt.subplot(2, 2, 3)
plt.hist(get_all_accuracy(), 50, density=True, facecolor='g', alpha=0.75)
plt.xlabel('F1')
plt.ylabel('Count')
plt.title('Histogram of Accuracy')
plt.grid(True)
plt.xlim((0.0, 1.0))
plt.axvline(x=best_harmonic_rep.runtime_properties['score'], color='red')



plt.subplot(2, 2, 4)
plt.hist(np.array(get_all_complexity())*-1, 50, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Complexity')
plt.ylabel('Count')
plt.title('Histogram of Complexity')
plt.grid(True)
plt.axvline(x=best_harmonic_rep.get_complexity()*-1, color='red')
plt.xlim((-6, -1))
plt.show()
