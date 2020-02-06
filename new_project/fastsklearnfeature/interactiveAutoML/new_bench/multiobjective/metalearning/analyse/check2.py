import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score

def print_constraints(features):
	print('acc: ' + str(features[0]) +
		  ' fair: ' + str(features[1]) +
		  ' k rel: ' + str(features[2]) +
		  ' k abs: ' + str(features[3]) +
		  ' robust: ' + str(features[4]) +
		  ' privacy: ' + str(features[5]))


'''
feature_list.append(hps['accuracy'])
			feature_list.append(hps['fairness'])
			feature_list.append(hps['k'])
			feature_list.append(hps['k'] * X_train.shape[1])
			feature_list.append(hps['robustness'])
			feature_list.append(cv_privacy)
			#differences to sample performance
			feature_list.append(cv_acc - hps['accuracy'])
			feature_list.append(cv_fair - hps['fairness'])
			feature_list.append(cv_k - hps['k'])
			feature_list.append((cv_k - hps['k']) * X_train.shape[1])
			feature_list.append(cv_robust - hps['robustness'])
			#privacy constraint is always satisfied => difference always zero => constant => unnecessary

			#metadata features
			feature_list.append(X_train.shape[0])#number rows
			feature_list.append(X_train.shape[1])#number columns
	'''

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


#print(logs_regression['features'])



X_train = logs_regression['features']
y_train = logs_regression['best_strategy']

meta_classifier = RandomForestClassifier(n_estimators=1000)
meta_classifier.fit(X_train, y_train)

scores = cross_val_score(meta_classifier, X_train, y_train, cv=5)

print(meta_classifier.feature_importances_)
#print(meta_classifier.score(X_test, y_test))





#get runbtime distributions

evo_id = 8
var_id = 1
chi2_id = 2
hyper_id = 7
weighted_ranking = 6

#get case where variance is way faster than evolution ...

dataset = logs_regression
runtime_dist = []

first_strategy = evo_id
second_strategy = chi2_id


mappnames = {1:'var', 2: 'chi2', 3:'acc rank', 4: 'robust rank', 5: 'fair rank', 6: 'weighted ranking', 7: 'hyperopt', 8: 'evo'}


run_v = []
strategy_v = []
runtime_v = []
for run in range(len(dataset['times_value'])):
	if dataset['features'][run][0] > 0.6:
		if logs_regression['best_strategy'][run] != 0:
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

print_constraints(dataset['features'][best_variance_against_evo])

success_check = np.zeros(9)
for run in range(len(dataset['success_value'])):
	for s in range(1, 9):
		if s in dataset['success_value'][run]:
			success_check[s] += np.sum(dataset['success_value'][run][s])



print_strategies(success_check / (len(dataset['success_value']*2)))

best_strategy_count = np.zeros(9)
for run in range(len(dataset['best_strategy'])):
	best_strategy_count[dataset['best_strategy'][run]] += 1

print("Best count: ")
print_strategies(best_strategy_count / len(dataset['best_strategy']))



