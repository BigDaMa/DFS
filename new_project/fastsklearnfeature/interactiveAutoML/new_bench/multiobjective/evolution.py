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
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy

import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
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
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from pymoo.factory import get_problem, get_termination
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time

from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.metrics import loss_sensitivity
from art.metrics import empirical_robustness
from sklearn.pipeline import FeatureUnion


from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier
from art.attacks import HopSkipJump


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.my_global_variable as my_global_variable



sensitive_attribute = "sex"

n_estimators = 5

df = pd.read_csv(Config.get('data_path') + '/adult/dataset_183_adult.csv', delimiter=',', header=0)
y = df['class']
del df['class']
X = df
one_hot = True

limit = 1000

X_train, X_test, y_train, y_test = train_test_split(X.values[0:limit,:], y.values[0:limit], test_size=0.5, random_state=42)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

continuous_columns = [0, 2, 4, 10, 11, 12]

xshape = X_train.shape[1]
if one_hot:
	ct = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False), [1,3,5,6,7,8,9,13])])
	scale = ColumnTransformer([("scale", MinMaxScaler(), continuous_columns)])

	pipeline = FeatureUnion([("o", ct),("s", scale)])

	X_train = pipeline.fit_transform(X_train)
	xshape = X_train.shape[1]
	print(xshape)
	X_test = pipeline.transform(X_test)

	print(ct.get_feature_names())


names = ct.get_feature_names()
for c in continuous_columns:
	names.append(str(X.columns[c]))

sensitive_ids = []
all_names = ct.get_feature_names()
for fname_i in range(len(all_names)):
	if all_names[fname_i].startswith('onehot__x6' + '_'):
		sensitive_ids.append(fname_i)


auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
fair_test = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_test[:, sensitive_ids[0]])


history = []

min_accuracy = 0.80
min_fairness = 0.90
min_robustness = 0.80
privacy_epsilon = None
max_number_features = X_train.shape[1]

my_global_variable.global_check = False


# define an objective function
def objective(features):
	model = Pipeline([
		('selection', MaskSelection(features)),
		('clf', LogisticRegression())
	])

	robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train,
								feature_selector=model.named_steps['selection'])
	robust_scorer_test = make_scorer(robust_score_test, greater_is_better=True, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
								feature_selector=model.named_steps['selection'])

	ncv = 5

	cv_acc = np.mean(
		cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=auc_scorer))
	cv_fair = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=fair_train))
	cv_robust = 1.0 - np.mean(cross_val_score(model, X_train, pd.DataFrame(y_train), cv=StratifiedKFold(ncv, random_state=42), scoring=robust_scorer))
	#cv_robust = 1.0

	print('cv acc: ' + str(cv_acc) + ' cv fair: ' + str(cv_fair) + ' cv robust: ' + str(cv_robust))

	if cv_acc > min_accuracy and cv_fair > min_fairness and cv_robust > min_robustness:
		model.fit(X_train, pd.DataFrame(y_train))
		test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
		test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
		test_robust = 1.0 - robust_scorer_test(model, X_test, pd.DataFrame(y_test))

		print('acc: ' + str(test_acc) + ' fair: ' + str(test_fair) + ' robust: ' + str(test_robust))

		if test_acc > min_accuracy and test_fair > min_fairness and test_robust > min_robustness:
			my_global_variable.global_check = True

			print("selected features: " + str(np.array(names)[features]))

	simplicity = -1 * np.sum(features)

	#change objectives
	#cv_acc = 1.0
	return [cv_acc, cv_fair, cv_robust, simplicity]

class MyProblem(Problem):

	def __init__(self):
		super().__init__(n_var=X_train.shape[1],
                         n_obj=3,
                         n_constr=0,
						 xl=0, xu=1, type_var=anp.bool)

	def _evaluate(self, x, out, *args, **kwargs):
		f1_all = []
		f2_all = []
		f3_all = []
		f4_all = []

		print(x)

		for i in range(len(x)):
			results = objective(x[i])
			f1_all.append(results[0]*-1)#accuracy
			f2_all.append(results[1]*-1)#fairness
			f3_all.append(results[2]*-1)#robustness
			f4_all.append(results[3] * -1)  # simplicity

			#g1_all.append(c1)


		out["F"] = anp.column_stack([f1_all, f2_all, f3_all])



problem = MyProblem()


population_size = 30
cross_over_rate = 0.9
algorithm = NSGA2(pop_size=population_size,
				  sampling=get_sampling("bin_random"),
				  crossover=get_crossover('bin_one_point'),#get_crossover("bin_hux"),#get_crossover("bin_two_point"),
				  mutation=BinaryBitflipMutation(1.0 / X_train.shape[1]),
				  elimate_duplicates=True,
				  #n_offsprings= cross_over_rate * population_size
				  )
'''
algorithm = NSGA2(pop_size=population_size,
				  sampling=get_sampling("bin_random"),
				  crossover=get_crossover("bin_hux"),#get_crossover("bin_two_point"),
				  mutation=get_mutation("bin_bitflip"),
				  elimate_duplicates=True)
'''

from pymoo.model.termination import Termination
class MyTermination(Termination):

	def __init__(self, my_constraint=None) -> None:
		super().__init__()
		self.my_constraint =np.array(my_constraint)*-1

	def _do_continue(self, algorithm):
		print('termination: ' + str(my_global_variable.global_check))
		if my_global_variable.global_check:
			F = algorithm.pop.get("F")

			for f_i in range(len(F)):
				if np.all(np.less(F[f_i], self.my_constraint)):
					return False

		return True

start_time = time.time()

res = minimize(problem=problem,
               algorithm=algorithm, termination=MyTermination([min_accuracy, min_fairness, min_robustness]),
               disp=False)

print("Best solution found: %s" % res.X.astype(np.int))
print("Function value: %s" % res.F)

print("needed time: " + str(time.time() - start_time))