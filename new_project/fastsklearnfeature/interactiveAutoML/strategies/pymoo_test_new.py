import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy

which_experiment = 'experiment3'#'experiment1'

numeric_representations: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/features.p", "rb"))

filtered = numeric_representations

'''
filtered = []
for f in numeric_representations:
	if isinstance(f, RawFeature):
		filtered.append(f)
	else:
		if isinstance(f.transformation, OneHotTransformation):
			filtered.append(f)
numeric_representations = filtered
'''

y_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_test.p", "rb")).values
#print(y_test)

my_names: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/names.p", "rb"))
print(my_names)

X_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/X_train.p", "rb"))

#print(X_train[:,7])

#todo: measure TP for each group and add objective
#todo: try misclassification constraint
#todo: restart search

bit2results= {}

foreigner = np.array(X_train[:,7])
gender = np.array(['female' in personal_status for personal_status in X_train[:,15]])

my_runner = Runner(c=1.0, sensitive=gender, labels=['bad', 'good'])
#my_runner = Runner(c=1.0, sensitive=foreigner, labels=['bad', 'good'])


history = []


# define an objective function
def objective(features):
	#_, test, pred_test, std_score, proba_pred_test = my_runner.run_pipeline(features, runs=1)
	results = my_runner.run_pipeline(features, runs=1)

	#print(features)

	'''
	assert len(y_test) == len(pred_test)
	which_observation_should_be_predicted_correctly = 333#131
	print(proba_pred_test[which_observation_should_be_predicted_correctly])

	true_class_index = -1
	for c_i in range(len(my_runner.pipeline.classes_)):
		if my_runner.pipeline.classes_[c_i] == y_test[which_observation_should_be_predicted_correctly]:
			true_class_index = c_i
			break

	print(proba_pred_test[which_observation_should_be_predicted_correctly][true_class_index])

	uncertainty_that_observation_is_classified_correctly = 1.0 - proba_pred_test[which_observation_should_be_predicted_correctly][true_class_index]

	print(str(pred_test[which_observation_should_be_predicted_correctly]) + ' ?= true: ' + str(y_test[which_observation_should_be_predicted_correctly]))

	#constraint_satisfied = pred_test[which_observation_should_be_predicted_correctly] == y_test[which_observation_should_be_predicted_correctly]
	constraint_satisfied = -proba_pred_test[which_observation_should_be_predicted_correctly][true_class_index] + 0.5
	print("constraint: " + str(constraint_satisfied))
	'''

	bit2results[tuple(features)] = [1.0 - results['auc'], results['complexity'], 1.0 - results['test_auc'], results['fair']]

	print(bit2results[tuple(features)])

	history.append(copy.deepcopy(results))
	pickle.dump(history, open("/tmp/evoltionary_feature_selection.p", "wb"))

	return results

class MyProblem(Problem):

	def __init__(self):
		super().__init__(n_var=len(numeric_representations),
                         n_obj=3,
                         n_constr=0, xl=0, xu=1, type_var=anp.bool)

	def _evaluate(self, x, out, *args, **kwargs):
		f1_all = []
		f2_all = []
		f3_all = []

		g1_all = []

		for i in range(len(x)):
			results = objective(x[i])
			f1_all.append(1.0 - results['auc'])
			f2_all.append(results['complexity'])
			f3_all.append(results['fair'])

			#g1_all.append(c1)

		#out["F"] = anp.column_stack([f1_all, f2_all, f3_all])
		out["F"] = anp.column_stack([f1_all, f2_all, f3_all])
		#out["F"] = anp.column_stack([f1_all, f2_all])
		#out["F"] = anp.column_stack([g1_all, f2_all])
		#out["G"] = anp.column_stack([g1_all])



problem = MyProblem()

'''
algorithm = GA(
    pop_size=10,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_hux"),
    mutation=get_mutation("bin_bitflip"),
    elimate_duplicates=True)
'''
#algorithm = NSGA2(pop_size=10, elimate_duplicates=True)
algorithm = NSGA2(pop_size=5,
				  sampling=get_sampling("bin_random"),
				  crossover=get_crossover("bin_hux"),#get_crossover("bin_two_point"),
				  mutation=get_mutation("bin_bitflip"),
				  elimate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               disp=False)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)

print("all:")
for i in range(len(res.X)):
	print(bit2results[tuple(res.X[i])])

acc = []
complexity = []
for element in res.F:
	acc.append(element[0])
	complexity.append(element[1])

complexity = np.array(complexity)
acc = np.array(acc)

ids = np.argsort(complexity)


plt.plot(complexity[ids], acc[ids])
plt.xlabel('Complexity')
plt.ylabel('Loss: 1.0 - AUC')
plt.show()

print('all all: ')
for _,v in bit2results.items():
	print(str(v) +',')

