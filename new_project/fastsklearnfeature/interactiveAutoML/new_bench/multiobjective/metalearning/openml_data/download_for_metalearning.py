import openml
from fastsklearnfeature.configuration.Config import Config
import pickle
import numpy as np

openml.config.apikey = Config.get('openML.apikey')


unique_data = {}


i = 0


for _, data_info in openml.datasets.list_datasets().items():
	if 'status' in data_info and data_info['status'] == 'active' \
			and 'NumberOfMissingValues' in data_info and data_info['NumberOfMissingValues'] == 0 \
			and 'NumberOfClasses' in data_info and data_info['NumberOfClasses'] == 2 \
			and 'NumberOfInstances' in data_info and data_info['NumberOfInstances'] > 150:

		try:
			if not data_info['name'] in unique_data:
				dataset = openml.datasets.get_dataset(data_info['did'])
				X, y, categorical_indicator, attribute_names = dataset.get_data(
					dataset_format='dataframe',
					target=dataset.default_target_attribute
				)
				print(np.unique(y))

				continuous_columns = dataset.get_features_by_type('numeric')
				categorical_features = dataset.get_features_by_type('nominal')
				i += 1
				print(data_info['name'])
				print(i)
			unique_data[data_info['name']] = data_info
			if i > 40:
				break
		except:
			pass

print(len(unique_data))

pickle.dump(list(unique_data.values()), open('/tmp/fitting_datasets.pickle', 'wb'))