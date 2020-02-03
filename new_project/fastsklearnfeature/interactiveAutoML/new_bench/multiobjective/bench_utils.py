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
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import unit_test_score

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import openml
import random

def get_data(data_path='/adult/dataset_183_adult.csv', continuous_columns = [0, 2, 4, 10, 11, 12], sensitive_attribute = "sex", limit = 1000):

	df = pd.read_csv(Config.get('data_path') + data_path, delimiter=',', header=0)
	y = df['class']
	del df['class']
	X = df
	one_hot = True

	X_train, X_test, y_train, y_test = train_test_split(X.values[0:limit,:], y.values[0:limit], test_size=0.5, random_state=42)

	sensitive_attribute_id = -1
	for c_i in range(len(df.columns)):
		if str(df.columns[c_i]) == sensitive_attribute:
			sensitive_attribute_id = c_i
			break


	categorical_features = list(set(list(range(X_train.shape[1]))) - set(continuous_columns))

	cat_sensitive_attribute_id = -1
	for c_i in range(len(categorical_features)):
		if categorical_features[c_i] == sensitive_attribute_id:
			cat_sensitive_attribute_id = c_i
			break

	xshape = X_train.shape[1]
	if one_hot:
		ct = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)])
		scale = ColumnTransformer([("scale", MinMaxScaler(), continuous_columns)])

		pipeline = FeatureUnion([("o", ct),("s", scale)])

		X_train = pipeline.fit_transform(X_train)
		xshape = X_train.shape[1]
		X_test = pipeline.transform(X_test)


	names = ct.get_feature_names()
	for c in continuous_columns:
		names.append(str(X.columns[c]))

	print(names)

	sensitive_ids = []
	all_names = ct.get_feature_names()
	for fname_i in range(len(all_names)):
		if all_names[fname_i].startswith('onehot__x' + str(cat_sensitive_attribute_id) + '_'):
			sensitive_ids.append(fname_i)


	#pickle.dump(names, open("/home/felix/phd/ranking_exeriments/names.p", "wb"))


	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	return X_train, X_test, y_train, y_test, names, sensitive_ids


def get_data_openml(data_infos, limit=None):
	found=False
	while not found:
		try:
			# pick random dataset
			data_id = random.randint(0, len(data_infos) - 1)
			dataset = openml.datasets.get_dataset(data_infos[data_id]['did'])
			X, y, categorical_indicator, attribute_names = dataset.get_data(
				dataset_format='dataframe',
				target=dataset.default_target_attribute
			)
			class_id = -1
			for f_i in range(len(dataset.features)):
				if dataset.features[f_i].name == dataset.default_target_attribute:
					class_id = f_i
					break

			continuous_columns = dataset.get_features_by_type('numeric')
			categorical_features = list(set(dataset.get_features_by_type('nominal')) - set([class_id]))

			# randomly draw one attribute as sensitive attribute from continuous attributes
			sensitive_attribute_id = categorical_features[random.randint(0, len(categorical_features) - 1)]

			found = True
		except:
			pass


	if type(limit) != type(None):
		X_train, X_test, y_train, y_test = train_test_split(X.values[0:limit,:], y.values[0:limit], test_size=0.5, random_state=42)
	else:
		X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.5, random_state=42)

	cat_sensitive_attribute_id = -1
	for c_i in range(len(categorical_features)):
		if categorical_features[c_i] == sensitive_attribute_id:
			cat_sensitive_attribute_id = c_i
			break

	ct = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)])
	scale = ColumnTransformer([("scale", MinMaxScaler(), continuous_columns)])

	pipeline = FeatureUnion([("o", ct), ("s", scale)])
	X_train = pipeline.fit_transform(X_train)
	X_test = pipeline.transform(X_test)

	names = ct.get_feature_names()
	for c in continuous_columns:
		names.append(str(attribute_names[c]))

	sensitive_ids = []
	all_names = ct.get_feature_names()
	for fname_i in range(len(all_names)):
		if all_names[fname_i].startswith('onehot__x' + str(cat_sensitive_attribute_id) + '_'):
			sensitive_ids.append(fname_i)

	le = preprocessing.LabelEncoder()
	le.fit(y_train)
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	return X_train, X_test, y_train, y_test, names, sensitive_ids