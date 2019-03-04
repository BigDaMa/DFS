import pickle
from typing import List, Dict, Any, Set
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
import numpy as np
import matplotlib.pyplot as plt

from fastsklearnfeature.reader.Reader import Reader
from fastsklearnfeature.splitting.Splitter import Splitter
from fastsklearnfeature.configuration.Config import Config
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats

def get_max_score_from_parent(candidate: CandidateFeature, name2score, start=False):
    max_score = -1
    if not start and str(candidate) in name2score:
        max_score = name2score[str(candidate)]
    for p in candidate.parents:
        max_score = max(max_score, get_max_score_from_parent(p, name2score, False))
    return max_score


# heart also raw features
file = '/home/felix/phd/logs_iteration_fast_feature/4iterations/all_data_iterations.p'


all_data = pickle.load(open(file, "rb"))


dataset_config = (Config.get('statlog_heart.csv'), int(Config.get('statlog_heart.target')))

s = Splitter(train_fraction=[0.6, 10000000], seed=42)

dataset = Reader(dataset_config[0], dataset_config[1], s)
raw_features = dataset.read()
X = dataset.splitted_values['train']

#delta mean -> avg, min, max gain





def calculate_MSE(candidate: CandidateFeature, X):
    ys = []
    for p in candidate.parents:
        p.fit(X)
        y = p.transform(X)

        ys.append(y)

    #correlation
    #score = np.corrcoef(np.matrix(ys[0]).A1, np.matrix(ys[1]).A1)[0,1]
    #score = stats.kendalltau(np.matrix(ys[0]).A1, np.matrix(ys[1]).A1)[0]
    #score = stats.spearmanr(np.matrix(ys[0]).A1, np.matrix(ys[1]).A1)[0]

    #score = (np.max(ys[0]) - np.min(ys[0])) - (np.max(ys[1]) - np.min(ys[1]))
    #score = np.abs(np.mean(ys[0]) - np.mean(ys[1]))

    #score = mean_squared_error(ys[0], ys[1])
    #score = r2_score(ys[0], ys[1])

    return score


def plot_range_analysis(operator_stddev_list: Dict[str, float], operator_gain_list: Dict[str, float]):
    for transformation, stddevs_transformation in operator_stddev_list.items():
        #if transformation == "nansum":
        ids = np.argsort(stddevs_transformation)
        plt.plot(np.array(stddevs_transformation)[ids], np.array(operator_gain_list[transformation])[ids], label=transformation)


        #plt.axvline(x=0.0, color='red')
        plt.axhline(y=0.0, color='green')


        plt.title(transformation)
        plt.xlabel("Metric")
        plt.ylabel("Gain")

        #plt.ylim(bottom=0)

        plt.legend()

        plt.show()



operator_stddev_list: Dict[str, float] = {}
operator_gain_list: Dict[str, float]  = {}

name2score: Dict[str, float] = {}


name2all_means: Dict[str, float] = {}

name2min: Dict[str, float] = {}
name2max: Dict[str, float] = {}


# count number of skipped transformations without gain if we knew the perfect threshold


for round in range(len(all_data)):

    remaining = all_data[round]

    for i in range(len(remaining)):
        if len(remaining[i]['candidate']) == 1:

            c: CandidateFeature = remaining[i]['candidate'][0]
            name2score[str(c)] = remaining[i]['score']
            if not isinstance(c, RawFeature) and remaining[i]['score'] >= 0:
                if len(c.parents) > 1:
                    #stddev_means = calculate_delta_mean(c, name2all_means, X)
                    #stddev_means = calculate_delta_mean(c, name2min, name2max, X)
                    stddev_means = calculate_MSE(c, X)


                    c: CandidateFeature = remaining[i]['candidate'][0]

                    if not c.transformation.name in operator_stddev_list:
                        operator_stddev_list[c.transformation.name] = []
                        operator_gain_list[c.transformation.name] = []

                    max_parent = get_max_score_from_parent(c, name2score, True)
                    if remaining[i]['score'] >= 0 and max_parent >= 0:
                        operator_gain_list[c.transformation.name].append(remaining[i]['score'] - max_parent)
                        operator_stddev_list[c.transformation.name].append(stddev_means)
                        print(str(c.transformation.name) + " stdddev: " + str(stddev_means) + " gain: " + str(remaining[i]['score'] - max_parent))

    plot_range_analysis(operator_stddev_list, operator_gain_list)

