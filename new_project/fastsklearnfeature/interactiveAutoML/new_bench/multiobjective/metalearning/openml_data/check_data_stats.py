import openml
import urllib.request

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


data_ids = [
					31,  # credit-g => personal status, foreign_worker
					1590,  # adult => sex, race
					1461,  # bank-marketing => age

					42193,#compas-two-years => sex, age, race
					1480,#ilpd => sex => V2
					804, #hutsof99_logis => age,gender
					42178,#telco-customer-churn => gender
					981, #kdd_internet_usage => gender
					40536, #SpeedDating => race
					40945, #Titanic => Sex
					451, #Irish => Sex
					945, #kidney => sex
					446, #prnn_crabs => sex
					1017, #arrhythmia => sex
					957, #braziltourism => sex
					41430, #DiabeticMellitus => sex
					1240, #AirlinesCodrnaAdult sex
					1018, #ipums_la_99-small
					55, #hepatitis
					802,#pbcseq
					38,#sick
					40713, #dis
					1003,#primary-tumor
					934, #socmob
					]

'''
for i in range(len(data_ids)):
	dataset = openml.datasets.get_dataset(data_ids[i], download_data=False)
	print(dataset.name + ': ' )
'''

map_dataset = {}

map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] = 'race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1,4}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'
map_dataset['42565'] = 'gender@{g1,g2}' #students
map_dataset['42132'] = 'race@{BLACKLIVESMATTER,OTHER}'#traffic



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

random_number=42

for key, value in map_dataset.items():
	with open(Config.get('data_path') + "/downloaded_arff/" + str(key) + ".arff") as f:
		df = a2p.load(f)

		print("dataset: " + str(key))


		name_dataset = openml.datasets.get_dataset(dataset_id=int(key), download_data=False).name

		number_instances = df.shape[0]
		number_attributes = df.shape[1]

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

		X_temp, X_test, y_temp, y_test = train_test_split(X_datat, y.values.astype('str'), test_size=0.2,
														  random_state=random_number, stratify=y.values.astype('str'))

		X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=0.25,
																		random_state=random_number, stratify=y_temp)

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
		X_validation = pipeline.transform(X_validation)
		X_test = pipeline.transform(X_test)

		number_features = X_train.shape[1]

		print(name_dataset + ": instances = " + str(number_instances) + " attributes = " + str(number_attributes) + " features = " + str(number_features))

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
		y_validation = le.transform(y_validation)
		y_test = le.transform(y_test)

		X_train_val = np.vstack((X_train, X_validation))
		y_train_val = np.append(y_train, y_validation)




