import pickle
from typing import List, Dict, Any, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
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





def get_max_score_from_parent(candidate: CandidateFeature, name2score, start=False):
    max_score = -1
    if not start and str(candidate) in name2score:
        max_score = name2score[str(candidate)]
    for p in candidate.parents:
        max_score = max(max_score, get_max_score_from_parent(p, name2score, False))
    return max_score




#understand how we can reduce number of representations by removing failing representations

name2score: Dict[str, float] = {}



operator_to_gain: Dict[str, float] = {}

for round in range(len(all_data)):

    remaining = all_data[round]

    for i in range(len(remaining)):
        if len(remaining[i]['candidate']) == 1:
            name2score[str(remaining[i]['candidate'][0])] = remaining[i]['score']
            c: CandidateFeature = remaining[i]['candidate'][0]
            if not isinstance(c, RawFeature):
                if not c.transformation.name in operator_to_gain:
                    operator_to_gain[c.transformation.name] = []

                max_parent = get_max_score_from_parent(c, name2score, True)
                if remaining[i]['score'] >= 0 and max_parent >= 0:
                    operator_to_gain[c.transformation.name].append(remaining[i]['score'] - max_parent)



for k, v in operator_to_gain.items():
    print(str(k) + ", avg, " + str(np.round(np.mean(v),2)) \
          + ",min, " + str(np.round(np.min(v),2)) \
          + ", max, " + str(np.round(np.max(v),2)) \
          + ", prob, " + str(np.round(np.sum(np.array(v) > 0.0) / float(len(v)),2)))

















