import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.time_measure import time_score
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import call

def print_constraints_2(features):
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
logs_regression = pickle.load(open('/home/felix/phd/meta_learn/random_configs/metalearning_data.pickle', 'rb'))
dataset = logs_regression

print(logs_regression['best_strategy'])
print(len(logs_regression['best_strategy']))

#print(logs_regression['features'])

my_score = make_scorer(time_score, logs=dataset)

X_train = logs_regression['features']
y_train = logs_regression['best_strategy']

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

scores = cross_val_score(meta_classifier, X_train, np.array(y_train) == 0, cv=10, scoring='f1')
print('did it fail: ' + str(np.mean(scores)))

scores = cross_val_score(meta_classifier, X_train, pd.DataFrame(y_train), cv=10, scoring=my_score)
#scores = cross_val_score(meta_classifier, X_train, pd.DataFrame(y_train), cv=10)
print(scores)
print("scores: " + str(np.mean(scores)))
#print(meta_classifier.feature_importances_)
#print(meta_classifier.score(X_test, y_test))



#print_constraints_2(dataset['features'][156])



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


for s in range(1, 9):
	with open('/tmp/' + mappnames[s] + '_success.txt', 'a') as the_file:
		for run in range(len(dataset['times_value'])):
			if s in dataset['success_value'][run] and len(dataset['success_value'][run][s]) >= 1:
				the_file.write(str(run) + '\n')



