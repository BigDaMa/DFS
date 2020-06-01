import numpy as np
import copy
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
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

class DataLoader(object):
	
	def __init__(self):
		self.map_dataset2name = {}
		self.map_dataset2name['31'] = 'German Credit'
		self.map_dataset2name['802'] = 'Primary Biliary Cirrhosis'
		self.map_dataset2name['1590'] = 'Adult'
		self.map_dataset2name['1461'] = 'Bank Marketing'
		self.map_dataset2name['42193'] = 'COMPAS'
		self.map_dataset2name['1480'] = 'Indian Liver Patient'
		# self.map_dataset2name['804'] = 'hutsof99_logis'
		self.map_dataset2name['42178'] = 'Telco Customer Churn'
		self.map_dataset2name['981'] = 'KDD Internet Usage'
		self.map_dataset2name['40536'] = 'Speed Dating'
		self.map_dataset2name['40945'] = 'Titanic'
		self.map_dataset2name['451'] = 'Irish Educational Transitions'
		# self.map_dataset2name['945'] = 'Kidney'
		self.map_dataset2name['446'] = 'Leptograpsus crabs'
		self.map_dataset2name['1017'] = 'Arrhythmia'
		self.map_dataset2name['957'] = 'Brazil Tourism'
		self.map_dataset2name['41430'] = 'Diabetic Mellitus'
		self.map_dataset2name['1240'] = 'AirlinesCodrnaAdult'
		self.map_dataset2name['1018'] = 'IPUMS Census'
		# self.map_dataset2name['55'] = 'Hepatitis'
		self.map_dataset2name['38'] = 'Thyroid Disease'
		self.map_dataset2name['1003'] = 'Primary Tumor'
		self.map_dataset2name['934'] = 'Social Mobility'
		
		self.map_name2id = {v: k for k, v in self.map_dataset2name.items()}

		self.map_dataset = {}

		self.map_dataset['31'] = 'foreign_worker@{yes,no}'
		self.map_dataset['802'] = 'sex@{female,male}'
		self.map_dataset['1590'] = 'sex@{Female,Male}'
		self.map_dataset['1461'] = 'AGE@{True,False}'
		self.map_dataset['42193'] = 'race_Caucasian@{0,1}'
		self.map_dataset['1480'] = 'V2@{Female,Male}'
		# self.map_dataset['804'] = 'Gender@{0,1}'
		self.map_dataset['42178'] = 'gender@STRING'
		self.map_dataset['981'] = 'Gender@{Female,Male}'
		self.map_dataset['40536'] = 'samerace@{0,1}'
		self.map_dataset['40945'] = 'sex@{female,male}'
		self.map_dataset['451'] = 'Sex@{female,male}'
		# self.map_dataset['945'] = 'sex@{female,male}'
		self.map_dataset['446'] = 'sex@{Female,Male}'
		self.map_dataset['1017'] = 'sex@{0,1}'
		self.map_dataset['957'] = 'Sex@{0,1,4}'
		self.map_dataset['41430'] = 'SEX@{True,False}'
		self.map_dataset['1240'] = 'sex@{Female,Male}'
		self.map_dataset['1018'] = 'sex@{Female,Male}'
		# self.map_dataset['55'] = 'SEX@{male,female}'
		self.map_dataset['38'] = 'sex@{F,M}'
		self.map_dataset['1003'] = 'sex@{male,female}'
		self.map_dataset['934'] = 'race@{black,white}'

	def get_data(self, dataset='Adult', random_number=42):

		if isinstance(dataset, str):
			dataset_key = self.map_name2id[dataset]
		else:
			dataset_key = str(dataset)

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
			key = list(self.map_dataset.keys())[random.randint(0, len(self.map_dataset) - 1)]

		value = self.map_dataset[key]
		with open(Config.get('data_path') + "/downloaded_arff/" + str(key) + ".arff") as f:
			df = a2p.load(f)

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

			#print(sensitive_attribute_id)

			X_datat = X.values
			for x_i in range(X_datat.shape[0]):
				for y_i in range(X_datat.shape[1]):
					if type(X_datat[x_i][y_i]) == type(None):
						if X.dtypes[y_i] == object:
							X_datat[x_i][y_i] = 'missing'
						else:
							X_datat[x_i][y_i] = np.nan

			X_temp, X_test, y_temp, y_test = train_test_split(X_datat, y.values.astype('str'), test_size=0.2,
															  random_state=random_number,
															  stratify=y.values.astype('str'))

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

			return X_train, X_validation, X_test, y_train, y_validation, y_test, names, sensitive_ids