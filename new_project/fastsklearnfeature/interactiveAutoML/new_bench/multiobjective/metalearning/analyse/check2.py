import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np

def print_constraints(features):
	print('acc: ' + str(features[0]) + ' fair: ' + str(features[1]) + ' k: ' + str(features[2]) + ' robust: ' + str(features[3]) + ' privacy: ' + str(features[4]))

def print_strategies(results):
	print("all strategies failed: " + str(results[0]) +
		  "\nvar rank: " + str(results[1]) +
		  '\nchi2 rank: ' + str(results[2]) +
		  '\naccuracy rank: ' + str(results[3]) +
		  '\nrobustness rank: ' + str(results[4]) +
		  '\nfairness rank: ' + str(results[5]) +
		  '\nweighted ranking: ' + str(results[6]) +
		  '\nhyperparameter opt: ' + str(results[7]) +
		  '\nevolution: ' + str(results[8])
		  )


#logs_adult = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_adult.pickle', 'rb'))
#logs_heart = pickle.load(open('/home/felix/phd/meta_learn/classification/metalearning_data_heart.pickle', 'rb'))
logs_regression = pickle.load(open('/home/felix/phd/meta_learn/cluster_openml/metalearning_data.pickle', 'rb'))


print(logs_regression['features'])




'''
print(logs_adult.keys())

X_train = logs_adult['features']
y_train = logs_adult['best_strategy']

X_test = logs_heart['features']
y_test = logs_heart['best_strategy']

meta_classifier = RandomForestClassifier(n_estimators=1000)
meta_classifier.fit(X_train, y_train)

print(meta_classifier.feature_importances_)
print(meta_classifier.score(X_test, y_test))

print(logs_adult['best_strategy'])
'''

print(logs_regression['best_strategy'])


#get runbtime distributions

evo_id = 8
var_id = 1
hyper_id = 7
weighted_ranking = 6

#get case where variance is way faster than evolution ...

dataset = logs_regression
runtime_dist = []

first_strategy = hyper_id
second_strategy = var_id
for run in range(len(dataset['avg_times'])):
	if dataset['avg_success'][run][first_strategy] == 1.0 and dataset['avg_success'][run][second_strategy] == 1.0:
		runtime_dist.append(dataset['avg_times'][run][first_strategy] - dataset['avg_times'][run][second_strategy])

print(max(runtime_dist))
best_variance_against_evo = np.argmax(runtime_dist)

#print_constraints(dataset['features'][best_variance_against_evo])


print(len(dataset['avg_success']))
print_strategies(np.sum(np.array(dataset['avg_success']), axis=0) / len(dataset['avg_success']))

best_strategy_count = np.zeros(9)
for run in range(len(dataset['best_strategy'])):
	best_strategy_count[dataset['best_strategy'][run]] += 1

print("Best count: ")
print_strategies(best_strategy_count / len(dataset['best_strategy']))



