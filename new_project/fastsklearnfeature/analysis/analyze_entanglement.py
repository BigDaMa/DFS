import pickle
from typing import List, Dict, Any, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
import numpy as np
import matplotlib.pyplot as plt

# heart also raw features
file = '/home/felix/phd/logs_iteration_fast_feature/4iterations/all_data_iterations.p'


all_data = pickle.load(open(file, "rb"))





def get_max_score_from_parent(candidate: CandidateFeature, name2score, start=False):
    max_score = -1
    if not start and str(candidate) in name2score:
        max_score = name2score[str(candidate)]
    for p in candidate.parents:
        max_score = max(max_score, get_max_score_from_parent(p, name2score, False))
    return max_score




#understand how we can reduce number of representations by removing failing representations

name2score: Dict[str, float] = {}

#count_combinations_improved = 0
#count_total = 0

for round in range(len(all_data)):

    count_combinations_improved = 0
    count_total = 0
    count_combinations_decreased_all = 0

    remaining = all_data[round]

    for i in range(len(remaining)):
        if len(remaining[i]['candidate']) == 1:
            name2score[str(remaining[i]['candidate'][0])] = remaining[i]['score']
        else:
            max_score = -1.0
            min_score = 2.0
            for t in range(len(remaining[i]['candidate'])):
                fname = remaining[i]['candidate'][t]
                max_score = max(max_score, name2score[str(fname)])
                min_score = min(min_score, name2score[str(fname)])
            if max_score < remaining[i]['score']:
                count_combinations_improved += 1
            if min_score > remaining[i]['score']:
                if remaining[i]['score'] >= 0:
                    count_combinations_decreased_all += 1
            count_total += 1


    try:
        print("Improved all: " + str(count_combinations_improved / float(count_total)))
        print("decreased all: " + str(count_combinations_decreased_all / float(count_total)))
    except:
        pass

















