import openml
from fastsklearnfeature.configuration.Config import Config
import pickle
import numpy as np
import random

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
			unique_data[data_info['name']] = data_info
			print(len(unique_data))
			if len(unique_data) > 40:
				break
		except:
			pass

print(len(unique_data))

pickle.dump(list(unique_data.values()), open('/tmp/fitting_datasets.pickle', 'wb'))