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

with open("/home/felix/phd/meta_learn/downloaded_arff/42178.arff") as f:
	df = a2p.load(f)
	print(df.columns)
	df['TotalCharges@REAL'] = pd.to_numeric(df['TotalCharges@STRING'], errors='coerce')
	df = df.drop(columns=['TotalCharges@STRING'])

	with open('/home/felix/phd/meta_learn/downloaded_arff/42178_new.arff', 'w') as ff:
		a2p.dump(df, ff)



