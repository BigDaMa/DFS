from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
import autograd.numpy as anp
from pymoo.model.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from pymoo.model.termination import Termination
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.cheating_global as cheating_global
import random
from pymoo.model.repair import Repair
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import openml
import random
from sklearn.impute import SimpleImputer
import copy
from sklearn.model_selection import StratifiedKFold


from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import hyperparameter_optimization
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.exhaustive import exhaustive
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_floating_selection

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.fabolas import run_fabolas

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fairness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import robustness_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf

from skrebate import ReliefF

from functools import partial



##start
map_dataset = {}

map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] ='race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
map_dataset['804'] = 'Gender@{0,1}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
map_dataset['945'] = 'sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1,4}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
map_dataset['55'] = 'SEX@{male,female}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['40713'] = 'SEX@{True,False}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'



from arff2pandas import a2p
import glob


number_instances = []
number_attributes = []
number_features = []


def get_class_attribute_name(df):
	for i in range(len(df.columns)):
		if str(df.columns[i]).startswith('class@'):
			return str(df.columns[i])

def get_sensitive_attribute_id(df, sensitive_attribute_name):
	for i in range(len(df.columns)):
		if str(df.columns[i]) == sensitive_attribute_name:
			return i

key= '1590'
value = map_dataset[key]
with open("/home/felix/phd/meta_learn/downloaded_arff/" + str(key) + ".arff") as f:
	df = a2p.load(f)

	print("dataset: " + str(key))

	number_instances.append(df.shape[0])
	number_attributes.append(df.shape[1])

	y = copy.deepcopy(df[get_class_attribute_name(df)])
	X = df.drop(columns=[get_class_attribute_name(df)])

	categorical_features = []
	continuous_columns = []
	for type_i in range(len(X.columns)):
		if X.dtypes[type_i] == object:
			categorical_features.append(type_i)
		else:
			continuous_columns.append(type_i)


	sensitive_attribute_id = get_sensitive_attribute_id(X, value)

	print(sensitive_attribute_id)

	X_datat = X.values
	for x_i in range(X_datat.shape[0]):
		for y_i in range(X_datat.shape[1]):
			if type(X_datat[x_i][y_i]) == type(None):
				if X.dtypes[y_i] == object:
					X_datat[x_i][y_i] = 'missing'
				else:
					X_datat[x_i][y_i] = np.nan


	limit = 200
	X_train, X_test, y_train, y_test = train_test_split(X_datat[0:limit,:], y.values[0:limit].astype('str'), test_size=0.5,
															random_state=42, stratify=y.values[0:limit].astype('str'))

	'''
	X_train, X_test, y_train, y_test = train_test_split(X_datat, y.values.astype('str'),
														test_size=0.5,
														random_state=42, stratify=y.values.astype('str'))
	'''

	cat_sensitive_attribute_id = -1
	for c_i in range(len(categorical_features)):
		if categorical_features[c_i] == sensitive_attribute_id:
			cat_sensitive_attribute_id = c_i
			break

	my_transformers = []
	if len(categorical_features) > 0:
		ct = ColumnTransformer(
			[("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)])
		my_transformers.append(("o", ct))
	if len(continuous_columns) > 0:
		scale = ColumnTransformer([("scale", Pipeline(
			[('impute', SimpleImputer(missing_values=np.nan, strategy='mean')), ('scale', MinMaxScaler())]),
									continuous_columns)])
		my_transformers.append(("s", scale))

	pipeline = FeatureUnion(my_transformers)
	pipeline.fit(X_train)
	X_train = pipeline.transform(X_train)
	X_test = pipeline.transform(X_test)

	number_features.append(X_train.shape[1])

	all_columns = []
	for ci in range(len(X.columns)):
		all_columns.append(str(X.columns[ci]).split('@')[0])
	X.columns = all_columns

	names = ct.get_feature_names()
	for c in continuous_columns:
		names.append(str(X.columns[c]))

	for n_i in range(len(names)):
		if names[n_i].startswith('onehot__x'):
			tokens = names[n_i].split('_')
			category = ''
			for ti in range(3, len(tokens)):
				category += '_' + tokens[ti]
			cat_id = int(names[n_i].split('_')[2].split('x')[1])
			names[n_i] = str(X.columns[categorical_features[cat_id]]) + category

	print(names)

	print(len(names))

	sensitive_ids = []
	all_names = ct.get_feature_names()
	for fname_i in range(len(all_names)):
		if all_names[fname_i].startswith('onehot__x' + str(cat_sensitive_attribute_id) + '_'):
			sensitive_ids.append(fname_i)

	print(sensitive_ids)

	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	cv_splitter = StratifiedKFold(5, random_state=42)

	'''
	backward_selection(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions=[], clf=LogisticRegression(), min_accuracy=1.0,
			  min_fairness=1.0, min_robustness=0.0, max_number_features=0.05, cv_splitter=cv_splitter)
	'''

	from skfeature.function.sparse_learning_based.MCFS import mcfs
	from skfeature.function.sparse_learning_based.ll_l21 import proximal_gradient_descent



	def my_mcfs(X, y):
		result =  mcfs(copy.deepcopy(X), X_train.shape[1])
		new_result = result.max(1)
		return new_result



	from sklearn.feature_selection import mutual_info_classif


	#rankings= [partial(model_score, estimator=ReliefF(n_neighbors=10))]
	#rankings = [variance]
	#rankings= [mutual_info_classif]
	#rankings = [my_mcfs]
	#weighted_ranking(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions=rankings,clf=LogisticRegression(), min_accuracy=1.0,min_fairness = 1.0, min_robustness = 0.0, max_number_features = 0.3, cv_splitter = cv_splitter)

	'''
	run_fabolas(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions=[],
					   clf=LogisticRegression(), min_accuracy=1.0,
					   min_fairness=1.0, min_robustness=0.0, max_number_features=0.05, cv_splitter=cv_splitter)
	'''

	backward_floating_selection(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions=[],
					 clf=LogisticRegression(), min_accuracy=1.0, min_fairness=1.0, min_robustness=0.0,
					 max_number_features=0.05, cv_splitter=cv_splitter)
