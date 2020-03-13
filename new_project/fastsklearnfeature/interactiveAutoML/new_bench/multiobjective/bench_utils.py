import numpy as np
import copy
from sklearn.pipeline import Pipeline
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
import random
from sklearn.impute import SimpleImputer
from arff2pandas import a2p

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



def get_fair_data1(dataset_key=None):
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

		sensitive_ids = []
		all_names = ct.get_feature_names()
		for fname_i in range(len(all_names)):
			if all_names[fname_i].startswith('onehot__x' + str(cat_sensitive_attribute_id) + '_'):
				sensitive_ids.append(fname_i)

		le = preprocessing.LabelEncoder()
		le.fit(y_train)
		y_train = le.fit_transform(y_train)
		y_test = le.transform(y_test)

		return X_train, X_test, y_train, y_test, names, sensitive_ids, key, sensitive_attribute_id

