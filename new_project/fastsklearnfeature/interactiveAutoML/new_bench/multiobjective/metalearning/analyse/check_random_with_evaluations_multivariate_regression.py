import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_recall
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import time_score2
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_avg_runtime
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import get_optimum_avg_runtime

from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
import copy
import glob



mappnames = {1:'var', 2: 'chi2', 3:'acc rank', 4: 'robust rank', 5: 'fair rank', 6: 'weighted ranking', 7: 'hyperopt', 8: 'evo'}

names = ['accuracy',
	 'fairness',
	 'k_rel',
	 'k',
	 'robustness',
	 'privacy',
	 'search_time',
	 'cv_acc - acc',
	 'cv_fair - fair',
	 'cv_k - k rel',
	 'cv_k - k',
	 'cv_robust - robust',
     'cv time',
	 'rows',
	 'columns']

def print_constraints_2(features):


	my_str = ''
	for i in range(len(names)):
		my_str += names[i] + ': ' + str(features[i]) + ' '
	print(my_str)


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



#get all files from folder

# list files "/home/felix/phd/meta_learn/random_configs_eval"
#all_files = glob.glob("/home/felix/phd/meta_learn/random_configs_eval_long/*.pickle")
#all_files = glob.glob("/home/felix/phd/meta_learn/random_configs_with_repair_optimization/*.pickle")
all_files = glob.glob("/home/felix/phd/meta_learn/combine_random_configs/*.pickle") #1hour
#all_files = glob.glob("/home/felix/phd/meta_learn/3h_configs/*.pickle") #1hour


dataset = {}
for afile in all_files:
	data = pickle.load(open(afile, 'rb'))
	for key in data.keys():
		if not key in dataset:
			dataset[key] = []
		dataset[key].extend(data[key])


print(dataset['best_strategy'])
print(len(dataset['best_strategy']))

print(dataset.keys())


#get maximum number of evaluations if a strategy is fastest
eval_strategies = []
for i in range(9):
	eval_strategies.append([])

print(eval_strategies)
for bests in range(len(dataset['best_strategy'])):
	current_best = dataset['best_strategy'][bests]
	if current_best > 0:
		eval_strategies[current_best].append(dataset['evaluation_value'][bests][current_best][0])

print(eval_strategies)

print("max evaluations:")
for i in range(9):
	if len(eval_strategies[i]) > 0:
		print(mappnames[i] + ' min evaluations: ' + str(np.min(eval_strategies[i])) + ' max evaluations: ' + str(np.max(eval_strategies[i])) + ' avg evaluations: ' + str(np.mean(eval_strategies[i])) + ' len evaluations: ' + str(len(eval_strategies[i])))











#print(logs_regression['features'])

my_score = make_scorer(time_score2, greater_is_better=False, logs=dataset)
my_recall_score = make_scorer(get_recall, logs=dataset)
my_runtime_score = make_scorer(get_avg_runtime, logs=dataset)
my_optimal_runtime_score = make_scorer(get_optimum_avg_runtime, logs=dataset)

X_train = dataset['features']
y_train = dataset['best_strategy']

meta_classifier = RandomForestClassifier(n_estimators=1000)
#meta_classifier = DecisionTreeClassifier(random_state=0, max_depth=3)
meta_classifier = meta_classifier.fit(X_train, np.array(y_train) == 0)
#meta_classifier = meta_classifier.fit(X_train, y_train)



'''
# Export as dot file
export_graphviz(meta_classifier, out_file='/tmp/tree.dot',
                feature_names = ['accuracy',
											  'fairness',
											  'k_rel',
											  'k',
											  'robustness',
											  'privacy',
											  'cv_acc - acc',
											  'cv_fair - fair',
											  'cv_k - k rel',
											  'cv_k - k',
											  'cv_robust - robust',
											  'rows',
											  'columns'],
                class_names = np.array(meta_classifier.classes_, dtype=str),#['nothing','var','chi2','acc rank','robust rank','fair rank','weighted ranking','hyperopt','evo'],#['success', 'failure'],
                rounded = True, proportion = False,
                precision = 2, filled = True)

call(['dot', '-Tpng', '/tmp/tree.dot', '-o', '/tmp/tree.png', '-Gdpi=600'])

plt.show()
'''



#meta_classifier = DummyClassifier(strategy="uniform")
#meta_classifier = DummyClassifier(strategy="most_frequent")
#meta_classifier = DummyClassifier(strategy="constant", constant=8)

#scores = cross_val_score(meta_classifier, X_train, np.array(y_train) == 0, cv=10, scoring='f1')
#print('did it fail: ' + str(np.mean(scores)))


success_ids = np.where(np.array(y_train) > 0)[0]
print(success_ids)


new_success_ids = []
for s_i in success_ids:
	delete_b = False
	for strategy_i in dataset['evaluation_value'][s_i].keys():
		if dataset['evaluation_value'][s_i][strategy_i][0] == 1:
			delete_b = True
			break
	if not delete_b:
		new_success_ids.append(s_i)

#success_ids = new_success_ids

print("training size: " + str(len(success_ids)))



#print(X_train)
X_data = np.array(X_train)[success_ids]
y_data = np.array(y_train)[success_ids]
groups = np.array(dataset['dataset_id'])[success_ids]

outer_cv_all = list(GroupKFold(n_splits=4).split(X_data, None, groups=groups))


#hyperparameter optimization
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


def get_runtime_for_fold_predictions(predictions, test_ids):
	all_runtimes = []
	for p_i in range(len(predictions)):
		current_strategy = predictions[p_i]
		current_id = success_ids[test_ids[p_i]]
		if current_strategy in dataset['times_value'][current_id] and len(
				dataset['times_value'][current_id][current_strategy]) >= 1:
			all_runtimes.append(min(dataset['times_value'][current_id][current_strategy]))
		else:
			all_runtimes.append(dataset['features'][current_id][6])
	return all_runtimes

def get_optimal_runtime_for_fold_predictions(test_ids):
	all_runtimes = []
	for p_i in range(len(test_ids)):
		current_id = success_ids[test_ids[p_i]]
		best_runtime = dataset['features'][current_id][6]
		for s in range(1, 9):
			if s in dataset['times_value'][current_id] and len(dataset['times_value'][current_id][s]) >= 1:
				runtime = min(dataset['times_value'][current_id][s])
				if runtime < best_runtime:
					best_runtime = runtime
		all_runtimes.append(best_runtime)
	return all_runtimes

def generate_target(train_ids):
	runtimes = np.zeros((len(train_ids), 8), dtype=np.float32)
	for c_i in range(8):
		current_strategy = c_i + 1
		for p_i in range(len(train_ids)):
			current_id = success_ids[train_ids[p_i]]
			if current_strategy in dataset['times_value'][current_id] and len(
					dataset['times_value'][current_id][current_strategy]) >= 1:
				runtimes[p_i, c_i] = min(dataset['times_value'][current_id][current_strategy])
			else:
				runtimes[p_i, c_i] = dataset['features'][current_id][6]
	return runtimes

avg_runtime_cv_scores = []


all_runtimes_in_cv_folds = []
strategies_in_cv_folds = []
optimal_in_cv_folds = []

for s_i in range(8):
	strategies_in_cv_folds.append([])

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse_score = make_scorer(mean_squared_error, greater_is_better=False, multioutput='uniform_average')
r2_scorer = make_scorer(r2_score, greater_is_better=True, multioutput='uniform_average')
#r2_scorer = make_scorer(r2_score, greater_is_better=True, multioutput='variance_weighted')

for train_ids, test_ids in outer_cv_all:
	print("train_ids: " + str(train_ids))
	print("test_ids: " + str(test_ids))
	inner_cv = GroupKFold(n_splits=4).split(X_data[train_ids, :], None, groups=groups[train_ids])

	rf = RandomForestRegressor()
	rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
								   cv=inner_cv, verbose=2, random_state=42,
								   n_jobs=-1,scoring=r2_scorer)
	# Fit the random search model
	rf_random.fit(X_data[train_ids, :], generate_target(train_ids))
	print(rf_random.best_params_)

	predicted_runtime = rf_random.predict(X_data[test_ids, :])
	best_strategy_per_instance = np.argmin(predicted_runtime, axis=1)
	best_strategy_per_instance = np.add(best_strategy_per_instance, 1)
	all_runtimes_in_cv_folds.extend(get_runtime_for_fold_predictions(best_strategy_per_instance, test_ids))

	for s_i in range(8):
		one_strategy = np.ones(len(best_strategy_per_instance)) * (s_i+1)
		strategies_in_cv_folds[s_i].extend(get_runtime_for_fold_predictions(one_strategy, test_ids))

	optimal_in_cv_folds.extend(get_optimal_runtime_for_fold_predictions(test_ids))
	


for s_i in range(8):
	print(mappnames[s_i+1] + ' cv' + " avg runtime: " + str(np.nanmean(strategies_in_cv_folds[s_i])) + " median runtime: " + str(np.nanmedian(strategies_in_cv_folds[s_i])) + ' std runtime: ' + str(np.nanstd(strategies_in_cv_folds[s_i])))

print('metalearning cv' + " avg runtime: " + str(np.nanmean(all_runtimes_in_cv_folds)) + " median runtime: " + str(np.nanmedian(all_runtimes_in_cv_folds)) + ' std runtime: ' + str(np.nanstd(all_runtimes_in_cv_folds)))
print('optimal cv' + " avg runtime: " + str(np.nanmean(optimal_in_cv_folds)) + " median runtime: " + str(np.nanmedian(optimal_in_cv_folds)) + ' std runtime: ' + str(np.nanstd(optimal_in_cv_folds)))


#print("Nested cv score - meta learning: " + str(np.mean(nested_cv_scores)))
#print("Nested squared cv score - meta learning: " + str(np.mean(nested_cv_squared_scores)))
#print("Nested Recall cv score - meta learning: " + str(np.mean(recall_cv_scores)))
#print("Nested avg runtime cv score - meta learning: " + str(np.mean(avg_runtime_cv_scores)))





#get runbtime distributions

evo_id = 8
var_id = 1
chi2_id = 2
hyper_id = 7
weighted_ranking = 6

#get case where variance is way faster than evolution ...

runtime_dist = []

first_strategy = evo_id
second_strategy = chi2_id

run_v = []
strategy_v = []
runtime_v = []
for run in range(len(dataset['times_value'])):
	if run in new_success_ids:
		if dataset['features'][run][0] > 0.6:
			if dataset['best_strategy'][run] != 0:
				for s in range(1, 9):
					run_v.append(run)
					strategy_v.append(mappnames[s])
					if s in dataset['times_value'][run] and len(dataset['times_value'][run][s]) >= 1:
						runtime_v.append(min(dataset['times_value'][run][s]))
					else:
						runtime_v.append(0.0)

df = pd.DataFrame(
    {'run': run_v,
     'strategy': strategy_v,
     'runtime': runtime_v
    })

sns.barplot(x="run", hue="strategy", y="runtime", data=df)
plt.show()





'''
labels = list(range(dataset['times_value']))
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()
'''





all_ids = []
for run in range(len(dataset['times_value'])):
	if first_strategy in dataset['success_value'][run] and np.sum(dataset['success_value'][run][first_strategy]) >= 1.0 and \
		second_strategy in dataset['success_value'][run] and np.sum(dataset['success_value'][run][second_strategy]) >= 1.0:
		#if dataset['features'][run][0] > 0.6:
		runtime_dist.append(min(dataset['times_value'][run][first_strategy]) - min(dataset['times_value'][run][second_strategy]))
		all_ids.append(run)

print(max(runtime_dist))
best_variance_against_evo = all_ids[np.argmax(runtime_dist)]


success_check = np.zeros(9)
s_names = []
for run in range(len(dataset['success_value'])):
	s_names = []
	for s in range(1, 9):
		s_names.append(mappnames[s])
		if s in dataset['success_value'][run]:
			success_check[s] += np.sum(dataset['success_value'][run][s])

success_v = success_check[1:] / (len(dataset['success_value']*2))

print(success_v)

x_id = np.arange(len(s_names))  # the label locations
width = 0.35  # the width of the bars
recall_ids = np.argsort(success_v)
print(recall_ids)
fig, ax = plt.subplots()

print(len(np.array(success_v)[recall_ids]))
print(len(x_id))

rects1 = ax.bar(x_id, np.array(success_v)[recall_ids], width)
ax.set_ylabel('Recall')
ax.set_title('Recall')
ax.set_xticks(x)
ax.set_xticklabels(np.array(s_names)[recall_ids])
fig.tight_layout()
plt.show()


print_strategies(success_check / (len(dataset['success_value']*2)))

best_strategy_count = np.zeros(9)
for run in range(len(dataset['best_strategy'])):

	best_strategy_count[dataset['best_strategy'][run]] += 1

print("Best count: ")
print_strategies(best_strategy_count / len(dataset['best_strategy']))

for s in range(1, 9):
	with open('/tmp/' + mappnames[s] + '_success.txt', 'w+') as the_file:
		for run in range(len(dataset['times_value'])):
			if s in dataset['success_value'][run] and dataset['success_value'][run][s][0] == True:
				the_file.write(str(run) + '\n')



