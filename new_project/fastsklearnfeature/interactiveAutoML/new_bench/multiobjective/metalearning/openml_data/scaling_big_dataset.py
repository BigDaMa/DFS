import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import numpy as np

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_mcfs
from sklearn.model_selection import train_test_split
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score


from fastsklearnfeature.configuration.Config import Config



from functools import partial
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

from sklearn.feature_selection import mutual_info_classif
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score
from skrebate import ReliefF
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global
import diffprivlib.models as models
from sklearn.model_selection import GridSearchCV

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import TPE
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import simulated_annealing
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.exhaustive import exhaustive
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.recursive_feature_elimination import recursive_feature_elimination

from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import random
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#static constraints: fairness, number of features (absolute and relative), robustness, privacy, accuracy

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data1
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from arff2pandas import a2p

#load list of viable datasets
data_infos = pickle.load(open(Config.get('data_path') + '/openml_data/fitting_datasets.pickle', 'rb'))

current_run_time_id = time.time()

time_limit = 60 * 60 * 3
n_jobs = 20
number_of_runs = 1

X_train_meta_classifier = []
y_train_meta_classifier = []

ranking_scores_info = []


acc_value_list = []
fair_value_list = []
robust_value_list = []
success_value_list = []
runtime_value_list = []
evaluation_value_list = []
k_value_list = []

dataset_did_list = []
dataset_sensitive_attribute_list = []

cv_splitter = StratifiedKFold(5, random_state=42)

auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)


def get_fair_datanew(dataset_key=None, number_observations=100):
	map_dataset = {}

	map_dataset['31'] = 'foreign_worker@{yes,no}'
	map_dataset['802'] = 'sex@{female,male}'
	map_dataset['1590'] = 'sex@{Female,Male}'
	map_dataset['1461'] = 'AGE@{True,False}'
	map_dataset['42193'] = 'race_Caucasian@{0,1}'
	map_dataset['1480'] = 'V2@{Female,Male}'
	# map_dataset['804'] = 'Gender@{0,1}'
	map_dataset['42178'] = 'gender@STRING'
	map_dataset['981'] = 'Gender@{Female,Male}'
	map_dataset['40536'] = 'samerace@{0,1}'
	map_dataset['40945'] = 'sex@{female,male}'
	map_dataset['451'] = 'Sex@{female,male}'
	# map_dataset['945'] = 'sex@{female,male}'
	map_dataset['446'] = 'sex@{Female,Male}'
	map_dataset['1017'] = 'sex@{0,1}'
	map_dataset['957'] = 'Sex@{0,1,4}'
	map_dataset['41430'] = 'SEX@{True,False}'
	map_dataset['1240'] = 'sex@{Female,Male}'
	map_dataset['1018'] = 'sex@{Female,Male}'
	# map_dataset['55'] = 'SEX@{male,female}'
	map_dataset['38'] = 'sex@{F,M}'
	map_dataset['1003'] = 'sex@{male,female}'
	map_dataset['934'] = 'race@{black,white}'


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

	key = dataset_key
	if type(dataset_key) == type(None):
		key = list(map_dataset.keys())[random.randint(0, len(map_dataset) - 1)]

	value = map_dataset[key]
	with open(Config.get('data_path') + "/downloaded_arff/" + str(key) + ".arff") as f:
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


		'''
		X_train, X_test, y_train, y_test = train_test_split(X_datat, y.values.astype('str'), test_size=0.5,
															random_state=42, stratify=y.values.astype('str'))
		'''
		X_train, X_test, y_train, y_test = train_test_split(X_datat[0:200,:], y.values[0:200].astype('str'), test_size=0.5,
															random_state=42, stratify=y.values[0:200].astype('str'))


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

		sensitive_ids = []
		all_names = ct.get_feature_names()
		for fname_i in range(len(all_names)):
			if all_names[fname_i].startswith('onehot__x' + str(cat_sensitive_attribute_id) + '_'):
				sensitive_ids.append(fname_i)

		le = preprocessing.LabelEncoder()
		le.fit(y_train)
		y_train = le.fit_transform(y_train)
		y_test = le.transform(y_test)

		#randomly draw features and randomly draw observations
		'''
		X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=number_observations,
															random_state=42, stratify=y_train)

		
		X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=number_observations,
												  random_state=42, stratify=y_test)
		'''

		return X_train, X_test, y_train, y_test, names, sensitive_ids, key, sensitive_attribute_id




X_train, X_test, y_train, y_test, names, sensitive_ids, data_did, sensitive_attribute_id = get_fair_datanew(dataset_key='1240')

mp_global.X_train = X_train
mp_global.X_test = X_test
mp_global.y_train = y_train
mp_global.y_test = y_test
mp_global.names = names
mp_global.sensitive_ids = sensitive_ids
mp_global.cv_splitter = cv_splitter

runs_per_dataset = 0
i = 1

results_heatmap = {}
i += 1

min_accuracy = 0.5
min_fairness = 0.0
min_robustness = 0.0
max_number_features = 1.0
max_search_time = 2 * 60
privacy = None

# Execute each search strategy with a given time limit (in parallel)
# maybe run multiple times to smooth stochasticity

model = LogisticRegression()
if type(privacy) != type(None):
	model = models.LogisticRegression(epsilon=privacy)
mp_global.clf = model

#define rankings
rankings = [variance,
			chi2_score_wo,
			fcbf,
			my_fisher_score,
			mutual_info_classif,
			my_mcfs]
#rankings.append(partial(model_score, estimator=ExtraTreesClassifier(n_estimators=1000))) #accuracy ranking
#rankings.append(partial(robustness_score, model=model, scorer=auc_scorer)) #robustness ranking
#rankings.append(partial(fairness_score, estimator=ExtraTreesClassifier(n_estimators=1000), sensitive_ids=sensitive_ids)) #fairness ranking
rankings.append(partial(model_score, estimator=ReliefF(n_neighbors=10)))  # relieff

strategy_name = ['TPE(Variance)',
			 'TPE($\chi^2$))',
			 'TPE(FCBF))',
			 'TPE(Fisher Score))',
			 'TPE(Mutual Information))',
			 'TPE(MCFS))',
			 'TPE(ReliefF))',
             'TPE(no ranking)',
             'Simulated Annealing(no ranking)',
			 'NSGA-II(no ranking)',
			 'Exhaustive Search(no ranking)',
			 'Forward Selection(no ranking)',
			 'Backward Selection(no ranking)',
			 'Forward Floating Selection(no ranking)',
			 'Backward Floating Selection(no ranking)',
			 'RFE(Logistic Regression)'
			]


mp_global.min_accuracy = min_accuracy
mp_global.min_fairness = min_fairness
mp_global.min_robustness = min_robustness
mp_global.max_number_features = max_number_features
mp_global.max_search_time = max_search_time

mp_global.configurations = []
#add single rankings
strategy_id = 1
for r in range(len(rankings)):
	for run in range(number_of_runs):
		configuration = {}
		configuration['ranking_functions'] = copy.deepcopy([rankings[r]])
		configuration['run_id'] = copy.deepcopy(run)
		configuration['main_strategy'] = weighted_ranking
		configuration['strategy_id'] = copy.deepcopy(strategy_id)
		mp_global.configurations.append(configuration)
	strategy_id +=1

main_strategies = [TPE,
				   simulated_annealing,
				   evolution,
				   exhaustive,
				   forward_selection,
				   backward_selection,
				   forward_floating_selection,
				   backward_floating_selection,
				   recursive_feature_elimination]

#run main strategies
for strategy in main_strategies:
	for run in range(number_of_runs):
			configuration = {}
			configuration['ranking_functions'] = []
			configuration['run_id'] = copy.deepcopy(run)
			configuration['main_strategy'] = copy.deepcopy(strategy)
			configuration['strategy_id'] = copy.deepcopy(strategy_id)
			mp_global.configurations.append(configuration)
	strategy_id += 1

print(mp_global.configurations)

def my_function(config_id):
	new_result = {}
	search_times = []
	successes = []
	number_runs = 1
	for run_i in range(number_runs):
		conf = mp_global.configurations[config_id]
		result = conf['main_strategy'](mp_global.X_train, mp_global.X_test, mp_global.y_train, mp_global.y_test, mp_global.names, mp_global.sensitive_ids,
					 ranking_functions=conf['ranking_functions'],
					 clf=mp_global.clf,
					 min_accuracy=mp_global.min_accuracy,
					 min_fairness=mp_global.min_fairness,
					 min_robustness=mp_global.min_robustness,
					 max_number_features=mp_global.max_number_features,
					 max_search_time=mp_global.max_search_time,
					 cv_splitter=mp_global.cv_splitter)
		new_result['strategy_id'] = copy.deepcopy(conf['strategy_id'])
		#new_result['strategy_name'] = copy.deepcopy(conf['ranking_functions'][0].__name__)
		successes.append(result['success'])
		search_times.append(result['time'])

	if np.sum(successes) == number_runs:
		new_result['success'] = True
		new_result['time'] = np.mean(search_times)
	else:
		new_result['success'] = False


	return new_result


results = []
check_strategies = np.zeros(strategy_id)
with ProcessPool(max_workers=16) as pool:
	future = pool.map(my_function, list(range(len(mp_global.configurations))), timeout=max_search_time)

	iterator = future.result()
	while True:
		try:
			result = next(iterator)
			if result['success'] == True:
				try:
					print("found: " + str(result['strategy_id']))
					print(result)
					results_heatmap [(min_accuracy, result['strategy_id'])] = (result['time'], result['strategy_id'])

				except:
					print("fail strategy Id: " + str(result['strategy_id']))
		except StopIteration:
			break
		except TimeoutError as error:
			print("function took longer than %d seconds" % error.args[1])
		except ProcessExpired as error:
			print("%s. Exit code: %d" % (error, error.exitcode))
		except Exception as error:
			print("function raised %s" % error)
			print(error.traceback)  # Python's traceback of remote process
print('my heat map is here: ' + str(results_heatmap))



