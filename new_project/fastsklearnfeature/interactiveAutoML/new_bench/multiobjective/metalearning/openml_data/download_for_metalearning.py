import openml
from fastsklearnfeature.configuration.Config import Config
import pickle
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from fastsklearnfeature.configuration.Config import Config
from sklearn import preprocessing
import openml
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

openml.config.apikey = Config.get('openML.apikey')


unique_data = {}


for _, data_info in openml.datasets.list_datasets().items():
	if 'status' in data_info and data_info['status'] == 'active' \
			and 'NumberOfMissingValues' in data_info and data_info['NumberOfMissingValues'] == 0 \
			and 'NumberOfClasses' in data_info and data_info['NumberOfClasses'] == 2 \
			and 'NumberOfInstances' in data_info and data_info['NumberOfInstances'] > 250:

		try:

			dataset = openml.datasets.get_dataset(data_info['did'])
			print(data_info)

			continuous_columns = []
			categorical_features = []
			sensitive_attribute_id = -1
			X, y, categorical_indicator, attribute_names = dataset.get_data(
					dataset_format='dataframe',
					target=dataset.default_target_attribute
				)

			class_id = -1
			for f_i in range(len(dataset.features)):
				if dataset.features[f_i].name == dataset.default_target_attribute:
					class_id = f_i
					break

			for f_i in range(len(attribute_names)):
				for data_feature_o in range(len(dataset.features)):
					if attribute_names[f_i] == dataset.features[data_feature_o].name:
						if dataset.features[data_feature_o].data_type == 'nominal':
							categorical_features.append(f_i)
						if dataset.features[data_feature_o].data_type == 'numeric':
							continuous_columns.append(f_i)
						break

			# randomly draw one attribute as sensitive attribute from continuous attributes
			sensitive_attribute_id = categorical_features[random.randint(0, len(categorical_features) - 1)]

			X_train, X_test, y_train, y_test = train_test_split(X.values, y.values.astype('str'), test_size=0.5,
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
				scale = ColumnTransformer([("scale", MinMaxScaler(), continuous_columns)])
				my_transformers.append(("s", scale))

			pipeline = FeatureUnion(my_transformers)
			pipeline.fit(X_train[0:100,:])

			unique_data[data_info['name']] = data_info
			print(len(unique_data))
			pickle.dump(list(unique_data.values()), open('/tmp/fitting_datasets.pickle', 'wb'))
		except:
			pass

print(len(unique_data))

pickle.dump(list(unique_data.values()), open('/tmp/fitting_datasets.pickle', 'wb'))