import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List


def create_min_curve(losses, test):
	min_losses = [losses[0]]
	test_equi = [test[0]]
	for l in np.arange(1, len(losses)):
		if min_losses[-1] > losses[l]:
			min_losses.append(losses[l])
			test_equi.append(test[l])
		else:
			min_losses.append(min_losses[-1])
			test_equi.append(test_equi[-1])
	return min_losses, test_equi

#all_features_200: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new2.p", "rb"))

#all_features_200: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_step.p", "rb"))

all_features_200: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new.p", "rb"))

#raw_first: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new.p", "rb"))

numeric_representations: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/features.p", "rb"))

name2feature = {}
name2id = {}
feature_names = []
for f in range(len(numeric_representations)):
	name2feature[str(numeric_representations[f])] = numeric_representations[f]
	name2id[str(numeric_representations[f])] = f
	feature_names.append(str(numeric_representations[f]))


losses = np.array([r['loss'] for r in all_features_200.results])
#losses_raw = [r['loss'] for r in raw_first.results]


complexities = np.zeros(len(losses))
feature_count = np.zeros(len(numeric_representations))

feature_scores = []

for t in range(len(numeric_representations)):
	feature_scores.append([])

for k,v in all_features_200.vals.items():
	for i in range(len(losses)):
		if v[i]:
			if k in name2feature:
				complexities[i] += name2feature[k].get_complexity()
				feature_count[name2id[k]] += 1
				feature_scores[name2id[k]].append(all_features_200.results[i]['loss'])

'''
test = [r['test'] for r in all_features_200.results]
test_raw = [r['test'] for r in raw_first.results]
'''

feature_score_average = []
feature_score_stddev = []
for trial_losses in feature_scores:
	feature_score_average.append(np.average(trial_losses))
	feature_score_stddev.append(np.std(trial_losses))


feature_score_average = np.array(feature_score_average)
feature_score_stddev = np.array(feature_score_stddev)

ids = np.argsort(feature_score_average)

fig, ax = plt.subplots()
barlist= plt.bar(range(len(numeric_representations)), feature_score_average[ids], yerr=feature_score_stddev[ids])

for f_id in ids:
	fname = feature_names[f_id]
	if fname in all_features_200.best_trial['misc']['vals'] and all_features_200.best_trial['misc']['vals'][fname][0]:
		barlist[f_id].set_color('red')
	else:
		barlist[f_id].set_color('blue')

plt.xticks(range(len(numeric_representations)), np.array(feature_names)[ids], rotation='vertical')
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)
plt.show()




ids = np.argsort(feature_count*-1)

fig, ax = plt.subplots()
barlist= plt.bar(range(len(numeric_representations)), feature_count[ids])


for f_id in ids:
	fname = feature_names[f_id]
	if fname in all_features_200.best_trial['misc']['vals'] and all_features_200.best_trial['misc']['vals'][fname][0]:
		barlist[f_id].set_color('red')
	else:
		barlist[f_id].set_color('blue')



plt.xticks(range(len(numeric_representations)), np.array(feature_names)[ids], rotation='vertical')
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)
plt.show()




test = 1.0 - np.array([r['test'] for r in all_features_200.results])

ids = np.argsort(complexities)

plt.plot(complexities[ids], losses[ids], color='red', label='cv error')
plt.plot(complexities[ids], test[ids], color='blue', label='test error')
plt.xlabel('Complexity')
plt.ylabel('Loss (1.0 - AUC)')

leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.show()


plt.plot(range(len(complexities)), complexities, color='red')
plt.xlabel('Trials')
plt.ylabel('Complexity')

leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.show()