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

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score

import diffprivlib.models as models
from sklearn import preprocessing
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_data

from fastsklearnfeature.interactiveAutoML.feature_selection.WeightedRankingSelection import WeightedRankingSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection



def map_hyper2vals(hyper):
	new_vals = {}
	for k, v in hyper.items():
		new_vals[k] = [v]
	return new_vals


loss_history = []

def hyperparameter_optimization(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions= [], clf=None, min_accuracy = 0.0, min_fairness = 0.0, min_robustness = 0.0, max_number_features = None, max_search_time=np.inf, cv_splitter = None):

	start_time = time.time()

	auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
	fair_train = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_train[:, sensitive_ids[0]])
	fair_test = make_scorer(true_positive_rate_score, greater_is_better=True, sensitive_data=X_test[:, sensitive_ids[0]])

	def f_clf1(hps):
		mask = np.zeros(len(hps), dtype=bool)
		for k, v in hps.items():
			mask[int(k.split('_')[1])] = v


		#repair number of features if neccessary
		max_k = max(int(max_number_features * X_train.shape[1]), 1)
		if np.sum(mask) > max_k:
			id_features_used = np.nonzero(mask)[0]  # indices where features are used
			np.random.shuffle(id_features_used)  # shuffle ids
			ids_tb_deactived = id_features_used[max_k:]  # deactivate features
			for item_to_remove in ids_tb_deactived:
				mask[item_to_remove] = False

		for mask_i in range(len(mask)):
			hps['f_' + str(mask_i)] = mask[mask_i]

		model = Pipeline([
			('selection', MaskSelection(mask)),
			('clf', LogisticRegression())
		])

		return model, hps

	def f_to_min1(hps):
		model, hps = f_clf1(hps)

		print("hyperopt: " + str(hps))

		if np.sum(model.named_steps['selection'].mask) == 0:
			return {'loss': 4, 'status': STATUS_OK, 'model': model, 'cv_fair': 0.0, 'cv_acc': 0.0, 'cv_robust': 0.0, 'cv_number_features': 1.0}



		robust_scorer = make_scorer(robust_score, greater_is_better=True, X=X_train, y=y_train, model=clf, feature_selector=model.named_steps['selection'], scorer=auc_scorer)

		cv = GridSearchCV(model, param_grid={'clf__C': [1.0]}, cv=cv_splitter,
						  scoring={'AUC': auc_scorer, 'Fairness': fair_train, 'Robustness': robust_scorer},
						  refit=False)
		cv.fit(X_train, pd.DataFrame(y_train))
		cv_acc = cv.cv_results_['mean_test_AUC'][0]
		cv_fair = 1.0 - cv.cv_results_['mean_test_Fairness'][0]
		cv_robust = 1.0 - cv.cv_results_['mean_test_Robustness'][0]

		cv_number_features = float(np.sum(model.named_steps['selection']._get_support_mask())) / float(len(model.named_steps['selection']._get_support_mask()))

		loss = 0.0
		if cv_acc >= min_accuracy and \
				cv_fair >= min_fairness and \
				cv_robust >= min_robustness:
			if min_fairness > 0.0:
				loss += (min_fairness - cv_fair)
			if min_accuracy > 0.0:
				loss += (min_accuracy - cv_acc)
			if min_robustness > 0.0:
				loss += (min_robustness - cv_robust)
		else:
			if min_fairness > 0.0 and cv_fair < min_fairness:
				loss += (min_fairness - cv_fair) ** 2
			if min_accuracy > 0.0 and cv_acc < min_accuracy:
				loss += (min_accuracy - cv_acc) ** 2
			if min_robustness > 0.0 and cv_robust < min_robustness:
				loss += (min_robustness - cv_robust) ** 2

		loss_history.append(loss)


		return {'loss': loss, 'status': STATUS_OK, 'model': model, 'cv_fair': cv_fair, 'cv_acc': cv_acc, 'cv_robust': cv_robust, 'cv_number_features': cv_number_features, 'updated_parameters': hps}

	space = {}
	for f_i in range(X_train.shape[1]):
		space['f_' + str(f_i)] = hp.randint('f_' + str(f_i), 2)

	cv_fair = 0
	cv_acc = 0
	cv_robust = 0
	cv_number_features = 1.0

	number_of_evaluations = 0

	trials = Trials()
	i = 1
	success = False
	while True:
		if time.time() - start_time > max_search_time:
			break
		fmin(f_to_min1, space=space, algo=tpe.suggest, max_evals=i, trials=trials)

		#update repair in database
		try:
			current_trial = trials.trials[-1]
			if type(current_trial['result']['updated_parameters']) != type(None):
				trials._dynamic_trials[-1]['misc']['vals'] = map_hyper2vals(current_trial['result']['updated_parameters'])
		except:
			print("found an error in repair")



		number_of_evaluations += 1

		cv_fair = trials.trials[-1]['result']['cv_fair']
		cv_acc = trials.trials[-1]['result']['cv_acc']
		cv_robust = trials.trials[-1]['result']['cv_robust']
		cv_number_features = trials.trials[-1]['result']['cv_number_features']

		if cv_fair >= min_fairness and cv_acc >= min_accuracy and cv_robust >= min_robustness and cv_number_features <= max_number_features:
			model = trials.trials[-1]['result']['model']

			model.fit(X_train, pd.DataFrame(y_train))

			test_acc = 0.0
			if min_accuracy > 0.0:
				test_acc = auc_scorer(model, X_test, pd.DataFrame(y_test))
			test_fair = 0.0
			if min_fairness > 0.0:
				test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
			test_robust = 0.0
			if min_robustness > 0.0:
				test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test, model=model.named_steps['clf'], feature_selector=model.named_steps['selection'], scorer=auc_scorer)

			if test_fair >= min_fairness and test_acc >= min_accuracy and test_robust >= min_robustness:
				print('fair: ' + str(min(cv_fair, test_fair)) + ' acc: ' + str(min(cv_acc, test_acc)) + ' robust: ' + str(min(test_robust, cv_robust)) + ' k: ' + str(cv_number_features))
				success = True
				break

		i += 1

	if not success:
		try:
			cv_fair = trials.best_trial['result']['cv_fair']
			cv_acc = trials.best_trial['result']['cv_acc']
			cv_robust = trials.best_trial['result']['cv_robust']
			cv_number_features = trials.best_trial['result']['cv_number_features']
		except:
			pass

	runtime = time.time() - start_time
	return {'time': runtime, 'success': success, 'cv_acc': cv_acc, 'cv_robust': cv_robust, 'cv_fair': cv_fair, 'cv_number_features': cv_number_features, 'cv_number_evaluations': number_of_evaluations}





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

	X_train, X_test, y_train, y_test = train_test_split(X_datat, y.values.astype('str'), test_size=0.5,
															random_state=42, stratify=y.values.astype('str'))

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

	hyperparameter_optimization(X_train, X_test, y_train, y_test, names, sensitive_ids, ranking_functions=[], clf=LogisticRegression(), min_accuracy=1.0,
			  min_fairness=1.0, min_robustness=0.0, max_number_features=0.1, cv_splitter=cv_splitter, max_search_time=60 * 60)

	print(loss_history)
