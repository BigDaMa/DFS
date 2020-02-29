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

for key, value in map_dataset.items():
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

		names = ct.get_feature_names()
		for c in continuous_columns:
			names.append(str(X.columns[c]))

		sensitive_ids = []
		all_names = ct.get_feature_names()
		for fname_i in range(len(all_names)):
			if all_names[fname_i].startswith('onehot__x' + str(cat_sensitive_attribute_id) + '_'):
				sensitive_ids.append(fname_i)

		le = preprocessing.LabelEncoder()
		le.fit(y_train)
		y_train = le.fit_transform(y_train)
		y_test = le.transform(y_test)


import numpy as np
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(number_instances, 50, density=True, facecolor='g', alpha=0.75)
plt.title('Instances')
plt.grid(True)
plt.show()
print("istances: " + str(number_instances))

n, bins, patches = plt.hist(number_attributes, 50, density=True, facecolor='g', alpha=0.75)
plt.title('Attributes')
plt.grid(True)
plt.show()
print("attributes: " + str(number_attributes))

n, bins, patches = plt.hist(number_features, 50, density=True, facecolor='g', alpha=0.75)
plt.title('Features')
plt.grid(True)
plt.show()
print("featurs: " + str(number_features))

'''
with open("/home/felix/phd/meta_learn/downloaded_arff/40713.arff") as f:
	df = a2p.load(f)
	print(df.columns)
	#df['V1@NUMERIC']
	df['SEX@{True,False}'] = df['sex@{0,1,2}'] == '1'
	df = df.drop(columns=['sex@{0,1,2}'])

	with open('/home/felix/phd/meta_learn/downloaded_arff/40713_new.arff', 'w') as ff:
		a2p.dump(df, ff)
'''


