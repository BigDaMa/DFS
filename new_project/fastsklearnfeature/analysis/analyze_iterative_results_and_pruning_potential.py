import pickle
from typing import List, Dict, Any, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


def plot_histogram(scores, round, title='found'):
    # the histogram of the data
    n, bins, patches = plt.hist(scores, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.title('Histogram of Accuracy across representations of iteration ' + str(round) + " for " + title)
    plt.grid(True)
    plt.show()


def plot_data(all_data_round, round, title='found'):
    scores = [r['score'] for r in all_data_round]
    print(str(round) + ": " + str(len(scores)))

    plot_histogram(scores, round, title)

# heart also raw features
file = '/home/felix/phd/logs_iteration_fast_feature/4iterations/all_data_iterations.p'


all_data = pickle.load(open(file, "rb"))


# create one histogram per iteration

for round in range(len(all_data)):
    scores = [r['score'] for r in all_data[round]]
    print(str(round) + ": " + str(len(scores)))

    #plot_histogram(scores, round)




def filter_failed(failed_features, all_data_round):
    remaining = []
    for i in range(len(all_data_round)):
        found_failed = False
        for t in range(len(all_data_round[i]['candidate'])):
            if str(all_data_round[i]['candidate'][t]) in failed_features:
                found_failed = True
                break
        if found_failed == False:
            remaining.append(all_data_round[i])
    return remaining


def remove_representation_if_feature_occurs_more_than_once(all_data_round):
    remaining = []
    for i in range(len(all_data_round)):
        features = set()
        for t in range(len(all_data_round[i]['candidate'])):
            if str(all_data_round[i]['candidate'][t]) in features:
                break
            features.add(str(all_data_round[i]['candidate'][t]))
        if len(features) == len(all_data_round[i]['candidate']):
            remaining.append(all_data_round[i])
    return remaining

def remove_representations_that_use_raw_attribute_more_than_once(all_data_round):
    remaining = []
    for i in range(len(all_data_round)):
        attributes_all = set()
        found = False
        for t in range(len(all_data_round[i]['candidate'])):
            c:CandidateFeature = all_data_round[i]['candidate'][t]
            attributes = c.get_raw_attributes()
            for a in attributes:
                if str(a) in attributes_all:
                    found = True
                    break
                else:
                    attributes_all.add(str(a))
        if found == False:
            remaining.append(all_data_round[i])
    return remaining


def get_difference(bigger, smaller):
    b_string = set([str(b['candidate']) for b in bigger])
    s_string = set([str(s['candidate']) for s in smaller])

    difference = b_string ^ s_string

    skipped = []
    for b in bigger:
        if str(b['candidate']) in difference:
            skipped.append(b)

    return skipped



def check_candidate_if_parents_failed(candidate: CandidateFeature, failed_candidates):
    sum = 0
    for p in candidate.parents:
        if str(p) in failed_candidates:
            return True
        else:
            sum += check_candidate_if_parents_failed(p, failed_candidates)
    return sum > 0



def skip_features_that_did_not_improve_accuracy(failed_candidates, all_data_round):
    remaining = []
    #check whether we are allowed to use all features
    for i in range(len(all_data_round)):
        sum = 0
        for t in range(len(all_data_round[i]['candidate'])):
            sum += check_candidate_if_parents_failed(all_data_round[i]['candidate'][t], failed_candidates)
        if sum == 0:
            remaining.append(all_data_round[i])
    return remaining


def skip_combinations_that_did_not_improve_accuracy(failed_combinations: Set[Set[str]], all_data_round):
    remaining = []
    #check whether we are allowed to use all features
    for i in range(len(all_data_round)):
        current_set = frozenset([str(f) for f in all_data_round[i]['candidate']])
        remain = True
        for combination in failed_combinations:
            if combination.issubset(current_set):
                remain=False
                break
        if remain:
            remaining.append(all_data_round[i])

    return remaining



def get_max_score_from_parent(candidate: CandidateFeature, name2score, start=False):
    max_score = -1
    if not start and str(candidate) in name2score:
        max_score = name2score[str(candidate)]
    for p in candidate.parents:
        max_score = max(max_score, get_max_score_from_parent(p, name2score, False))
    return max_score

def check_candidate_candidate(candidate: CandidateFeature, score, name2score):
    if get_max_score_from_parent(candidate, name2score, True) > score:
        return False
    return True





#understand how we can reduce number of representations by removing failing representations

name2score: Dict[str, float] = {}

failed_features = set()
failed_candidates = set()
failed_combinations = set()

for round in range(len(all_data)):

    remaining = all_data[round]

    #remaining = filter_failed(failed_features, remaining)
    #remaining = remove_representation_if_feature_occurs_more_than_once(remaining)

    #aggressive
    #remaining = remove_representations_that_use_raw_attribute_more_than_once(remaining)
    remaining = skip_features_that_did_not_improve_accuracy(failed_candidates, remaining)
    remaining = skip_combinations_that_did_not_improve_accuracy(failed_combinations, remaining)


    skipped = get_difference(all_data[round], remaining)

    #print([str(s['candidate']) for s in skipped])


    #plot_data(remaining, round)
    #plot_data(skipped, round, title='skipped')


    print("saved runtime (min): " + str(np.sum([(r['time'] / 60.0) for r in skipped])))


    for i in range(len(remaining)):
        if len(remaining[i]['candidate']) == 1:
            name2score[str(remaining[i]['candidate'][0])] = remaining[i]['score']


        if remaining[i]['score'] < 0 and len(remaining[i]['candidate']) == 1:
            failed_features.add(str(remaining[i]['candidate'][0]))
    #print(failed_features)

    for i in range(len(remaining)):
        if len(remaining[i]['candidate']) == 1:
            if check_candidate_candidate(remaining[i]['candidate'][0], remaining[i]['score'], name2score) == False:
                failed_candidates.add(str(remaining[i]['candidate'][0]))

        if len(remaining[i]['candidate']) > 1:
            original_scores = []
            for t in range(len(remaining[i]['candidate'])):
                if str(remaining[i]['candidate'][t]) in name2score:
                    original_scores.append(name2score[str(remaining[i]['candidate'][t])])
                else:
                    break
            if len(original_scores) == len(remaining[i]['candidate']):
                failed_combinations.add(frozenset([str(f) for f in remaining[i]['candidate']]))


    #print(failed_candidates)
    #print(len(failed_candidates))










