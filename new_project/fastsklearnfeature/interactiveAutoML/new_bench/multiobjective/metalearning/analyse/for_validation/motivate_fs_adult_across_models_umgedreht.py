import pickle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

mappnames = {1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher)',
			 5: 'TPE(MIM)',
			 6: 'TPE(MCFS)',
			 7: 'TPE(ReliefF)',
			 8: 'TPE(NR)',
             9: 'SA(NR)',
			 10: 'NSGA-II(NR)',
			 11: 'ES(NR)',
			 12: 'SFS(NR)',
			 13: 'SBS(NR)',
			 14: 'SFFS(NR)',
			 15: 'SBFS(NR)',
			 16: 'RFE(LR)',
			 17: 'Complete Set'
			 }

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




#experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")

#experiment_folders = glob.glob("/home/felix/phd/versions_dfs/new_experiments/*/")
#experiment_folders = glob.glob("/home/felix/phd2/experiments_restric/*/")

#experiment_folders = glob.glob("/home/neutatz/data/experiments_rf/*/")
#experiment_folders = glob.glob("/home/felix/phd2/experiments_rf/*/")

#experiment_folders = glob.glob("/home/felix/phd2/expadult/*/")

#experiment_folders = glob.glob("/home/felix/phd2/experiments_new-dt/*/")

#experiment_folders = glob.glob("/home/felix/phd2/adult_1h/adult_filtered/*/")

experiment_folders = glob.glob("/home/felix/phd2/new_experiments_maybe_final/*/")

print(experiment_folders)


dataset = {}
dataset['best_strategy'] = []
dataset['validation_satisfied'] = []


dataset['success_value'] = []
dataset['success_value_validation'] = []
dataset['times_value'] = []
dataset['max_search_time'] = []

dataset['distance_to_test_constraint'] = []


def load_pickle(fname):
	data = []
	with open(fname, "rb") as f:
		while True:
			try:
				data.append(pickle.load(f))
			except EOFError:
				break
	return data


def is_successfull_validation_and_test(exp_results):
	return len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1]['success_test'] == True #also on test satisfied

def is_successfull_validation(exp_results):
	return len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[-1]  # constraints were satisfied on validation set



def distance_to_constraints_on_test(exp_results, best_run, info_dict):
	min_fairness = info_dict['constraint_set_list']['fairness']
	min_accuracy = info_dict['constraint_set_list']['accuracy']
	min_robustness = info_dict['constraint_set_list']['robustness']
	max_number_features = info_dict['constraint_set_list']['k']

	if type(best_run) != type(None):
		test_fair = exp_results[best_run]['test_fair']
		test_acc = exp_results[best_run]['test_acc']
		test_robust = exp_results[best_run]['test_robust']
		test_number_features = exp_results[best_run]['cv_number_features']
	else:
		test_fair = 0.0
		test_acc = 0.0
		test_robust = 0.0
		test_number_features = 1.0



	loss = 0.0
	if min_fairness > 0.0 and test_fair < min_fairness:
		loss += (min_fairness - test_fair) ** 2
	if min_accuracy > 0.0 and test_acc < min_accuracy:
		loss += (min_accuracy - test_acc) ** 2
	if min_robustness > 0.0 and test_robust < min_robustness:
		loss += (min_robustness - test_robust) ** 2
	if max_number_features < 1.0 and test_number_features > max_number_features:
		loss += (test_number_features - max_number_features) ** 2

	return loss

def distance_to_constraints_on_validation(exp_results, best_run, info_dict):
	min_fairness = info_dict['constraint_set_list']['fairness']
	min_accuracy = info_dict['constraint_set_list']['accuracy']
	min_robustness = info_dict['constraint_set_list']['robustness']
	max_number_features = info_dict['constraint_set_list']['k']

	if type(best_run) != type(None):
		test_fair = exp_results[best_run]['cv_fair']
		test_acc = exp_results[best_run]['cv_acc']
		test_robust = exp_results[best_run]['cv_robust']
		test_number_features = exp_results[best_run]['cv_number_features']
	else:
		test_fair = 0.0
		test_acc = 0.0
		test_robust = 0.0
		test_number_features = 1.0



	loss = 0.0
	if min_fairness > 0.0 and test_fair < min_fairness:
		loss += (min_fairness - test_fair) ** 2
	if min_accuracy > 0.0 and test_acc < min_accuracy:
		loss += (min_accuracy - test_acc) ** 2
	if min_robustness > 0.0 and test_robust < min_robustness:
		loss += (min_robustness - test_robust) ** 2
	if max_number_features < 1.0 and test_number_features > max_number_features:
		loss += (test_number_features - max_number_features) ** 2

	return loss


strategy_distance_test = {}
strategy_distance_validation = {}
for s in range(1, len(mappnames) + 1):
	strategy_distance_test[s] = []
	strategy_distance_validation[s] = []


number_ml_scenarios = 50000

my_latex_fair = ""
my_latex_safety = ""
my_latex_privacy = ""
my_latex_complexity = ""


latex_model = {}

name2color = {}
name2color['Logistic Regression'] = 'color2'
name2color['Decision Tree'] = 'color4'
name2color['Gaussian Naive Bayes'] = 'color8'
name2color['Random Forest'] = 'color16'

name2marker = {}
name2marker['Logistic Regression'] = 'o'
name2marker['Decision Tree'] = 'star'
name2marker['Gaussian Naive Bayes'] = 'triangle'
#name2marker['Random Forest'] = 'color16'

name2color = {}
name2color[0] = 'color2'
name2color[1] = 'color4'
name2color[2] = 'color8'
name2color['Random Forest'] = 'color16'

name2marker = {}
name2marker[0] = 'o'
name2marker[1] = 'star'
name2marker[2] = 'triangle'

max_number_features = {}


name2int= {}
name2int['Logistic Regression'] = 0
name2int['Gaussian Naive Bayes'] = 1
name2int['Decision Tree'] = 2

#for model_name in ['Logistic Regression', 'Gaussian Naive Bayes', 'Decision Tree']:
for model_name in [0,1,2]:
#for model_name in ['Decision Tree']:

	latex_model[model_name] = '''\\nextgroupplot['''

	if model_name == 2:
		latex_model[model_name] += 'xlabel=Fairness, '


	latex_model[model_name] += '''ylabel=F1 Score, xmin=0, xmax=1,legend to name=testLegendlabelsMotivation,legend entries={
	Logistic Regression (LR), 
	Naive Bayes (NB), 
	Decision Tree (DT),
	Original Feature Set} ] 
	
	\\addlegendimage{only marks, color=color2, mark=o}
	\\addlegendimage{only marks, color=color4, mark=star}
	\\addlegendimage{only marks, color=color8, mark=triangle}
	\\addlegendimage{only marks, color=black, mark=*}
	'''

	datapoints = []

	oracle_distance_validation = []
	oracle_distance_test = []
	run_count = 0
	for efolder in experiment_folders:
		run_folders = sorted(glob.glob(efolder + "*/"))
		for rfolder in run_folders:
			try:
				info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))

				if info_dict['dataset_id'] == '42193' and name2int[info_dict['constraint_set_list']['model']] == model_name:
					print(info_dict)
					print(info_dict['constraint_set_list']['model'])

					distance_of_this_run_test = []
					distance_of_this_run_validation = []
					for s in range(1, len(mappnames) + 1):
						exp_results = []
						try:
							exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')

							print(exp_results)
						except:
							pass

						print(len(exp_results))


						for min_r in range(len(exp_results)):
							try:
								validation_fair = exp_results[min_r]['cv_fair']
								validation_acc = exp_results[min_r]['cv_acc']
								validation_robust = exp_results[min_r]['cv_robust']
								validation_number_features = exp_results[min_r]['cv_number_features']

								print([validation_fair, validation_acc, validation_robust, validation_number_features])

								validation_time = 0.0
								if True:#exp_results[min_r]['success_test']:
									validation_time = exp_results[min_r]['time']


								validation_privacy = info_dict['constraint_set_list']['privacy']
								if type(validation_privacy) == type(None):
									validation_privacy = 0.0
									break

								if validation_number_features == 1.0:
									max_number_features[model_name] = [validation_fair, validation_acc, validation_robust, validation_number_features]

								datapoints.append([validation_fair, validation_acc, validation_robust, validation_number_features, validation_privacy, validation_time])

								if s == 17:
									print(exp_results)
									print([validation_fair, validation_acc, validation_robust, validation_number_features, validation_privacy])


								test_fair = exp_results[min_r]['test_fair']
								test_acc = exp_results[min_r]['test_acc']
								test_robust = exp_results[min_r]['test_robust']

								#datapoints.append([test_fair, test_acc, test_robust, validation_number_features,validation_privacy])


							except:
								pass



				run_count += 1
			except FileNotFoundError:
				pass
			if run_count == number_ml_scenarios:
				break
		if run_count == number_ml_scenarios:
			break


	datapoints = np.array(datapoints)

	'''
	plt.scatter(datapoints[:,1], datapoints[:,0], marker="x")
	plt.axhline(y=0.8134917452619665, color='red')
	plt.axvline(x=0.5, color='green')
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.xlabel('Validation Accuracy')
	plt.ylabel('Validation Fairness')
	plt.show()
	'''

	my_latex_fair += '\\nextgroupplot['
	if model_name == 0:
		my_latex_fair += 'ylabel=Fairness,'
	my_latex_fair += '] '

	my_tuple_set = set()
	my_latex_fair += '\\addplot[only marks, color=' + str(name2color[model_name]) + ', mark=' + str(name2marker[model_name]) + '] coordinates {'
	for i in range(len(datapoints)):
		current_tuple = (datapoints[i,0], datapoints[i,1],)
		if not current_tuple in my_tuple_set:
			my_tuple_set.add(current_tuple)
			my_latex_fair += '(' + str(datapoints[i,1]) + ',' + str(datapoints[i,0]) + ')'
	my_latex_fair += '};\n'

	my_latex_fair += '\\addplot[only marks, color=black, mark=*, mark options={solid, scale=2}] coordinates {(' + str(max_number_features[model_name][1]) + ',' + str(max_number_features[model_name][0]) + ')};\n\n'


	print("len: " + str(len(datapoints)) + " vs " + str(len(my_tuple_set)))

	'''
	plt.scatter(datapoints[:,1], datapoints[:,2], marker="x")
	plt.axhline(y=0.3755500160519706, color='red')
	plt.axvline(x=0.5, color='green')
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.xlabel('Validation Accuracy')
	plt.ylabel('Validation Robustness')
	plt.show()
	'''

	my_latex_safety += '\\nextgroupplot['
	if model_name == 0:
		my_latex_safety += 'ylabel=Safety,'
	my_latex_safety += '] '

	my_tuple_set = set()
	my_latex_safety += '\\addplot[only marks, color=' + str(name2color[model_name]) + ', mark=' + str(name2marker[model_name]) + '] coordinates {'
	for i in range(len(datapoints)):
		current_tuple = (datapoints[i, 2], datapoints[i, 1],)
		if not current_tuple in my_tuple_set:
			my_tuple_set.add(current_tuple)
			my_latex_safety += '(' + str(datapoints[i,1]) + ',' + str(datapoints[i,2]) + ')'
	my_latex_safety += '};\n'

	my_latex_safety += '\\addplot[only marks, color=black, mark=*, mark options={solid, scale=2}] coordinates {(' + str(max_number_features[model_name][1]) + ',' + str(max_number_features[model_name][2]) + ')};\n\n'


	#print(my_latex)

	'''
	plt.scatter(datapoints[:,1], datapoints[:,3], marker="x")
	plt.xlim((0,1))
	plt.ylim((0,1))
	plt.xlabel('Validation Accuracy')
	plt.ylabel('Feature Complexity')
	plt.show()

	plt.scatter(datapoints[:,1], datapoints[:,4], marker="x")
	plt.xlim((0,1))
	plt.yscale('log')
	plt.xlabel('Validation Accuracy')
	plt.ylabel('Differential Privacy Epsilon')
	plt.show()
	'''

	my_tuple_set = set()
	my_latex_privacy += '\\addplot[only marks, color=' + str(name2color[model_name]) + ', mark=' + str(name2marker[model_name]) + '] coordinates {'
	for i in range(len(datapoints)):
		current_tuple = (datapoints[i, 4], datapoints[i, 1],)
		if not current_tuple in my_tuple_set:
			my_tuple_set.add(current_tuple)
			my_latex_privacy += '(' + str(datapoints[i,4]) + ',' + str(datapoints[i,1]) + ')'
	my_latex_privacy += '};\n'

	'''
	plt.scatter(datapoints[:,1], datapoints[:,5], marker="x")
	plt.xlim((0,1))
	plt.xlabel('Validation Accuracy')
	plt.ylabel('Search Time')
	plt.show()
	'''

	latex_model[model_name] += '\\nextgroupplot['
	if model_name == 2:
		latex_model[model_name] += 'xlabel=Feature Set Size,'
	latex_model[model_name] += 'xmin=0, xmax=1] '

	my_tuple_set = set()
	my_latex_complexity = '\\addplot[only marks, color=' + str(name2color[model_name]) + ', mark=' + str(name2marker[model_name]) + '] coordinates {'
	for i in range(len(datapoints)):
		current_tuple = (datapoints[i, 3], datapoints[i, 1],)
		if not current_tuple in my_tuple_set:
			my_tuple_set.add(current_tuple)
			my_latex_complexity += '(' + str(datapoints[i,3]) + ',' + str(datapoints[i,1]) + ')'
	my_latex_complexity += '};\n'

	latex_model[model_name] += my_latex_complexity
	latex_model[model_name] += '\\addplot[only marks, color=black, mark=*, mark options={solid, scale=2}] coordinates {(' + str(max_number_features[model_name][3]) + ',' + str(max_number_features[model_name][1]) + ')};\n\n'


print(my_latex_fair)

print(my_latex_safety)
