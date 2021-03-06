import pandas as pd
from typing import List
from fastsklearnfeature.candidates.RawFeature import RawFeature

class ScikitReader:
	def __init__(self, X_train, y_train, feature_names: List[str]=None, feature_is_categorical: List[bool]=None):
		self.X_train = X_train
		self.y_train = y_train
		self.raw_features: List[RawFeature] = []
		self.feature_names = feature_names
		self.feature_is_categorical = feature_is_categorical

		if type(self.feature_names) != type(None):
			assert len(self.feature_names) == self.X_train.shape[1]
		if type(self.feature_is_categorical) != type(None):
			assert len(self.feature_is_categorical) == self.X_train.shape[1]

	def read(self) -> List[RawFeature]:

		self.dataframe = pd.DataFrame(data=self.X_train, columns=self.feature_names)


		if type(self.feature_is_categorical) != type(None):
			for feature_id in range(len(self.feature_is_categorical)):
				if not self.feature_is_categorical[feature_id]:
					self.dataframe[self.dataframe.columns[feature_id]] = pd.to_numeric(self.dataframe[self.dataframe.columns[feature_id]])

		if type(self.feature_is_categorical) == type(None):
			for feature_id in range(len(self.dataframe.columns)):
				try:
					self.dataframe[self.dataframe.columns[feature_id]] = pd.to_numeric(self.dataframe[self.dataframe.columns[feature_id]])
				except:
					pass

		self.splitted_values = {}
		self.splitted_target = {}

		self.splitted_target['train'] = self.y_train
		self.splitted_target['test'] = []
		self.splitted_values['train'] = self.X_train
		self.splitted_values['test'] = []

		for attribute_i in range(self.dataframe.shape[1]):
			feature_name = 'Feature' + str(self.dataframe.columns[attribute_i])
			if type(self.feature_names) != type(None):
				feature_name = self.feature_names[attribute_i]


			rf = RawFeature(feature_name, attribute_i, {})
			rf.derive_properties(self.dataframe[self.dataframe.columns[attribute_i]].values)
			if not rf.is_numeric():
				rf.properties['categorical'] = True
			if type(self.feature_is_categorical) != type(None):
				rf.properties['categorical'] = self.feature_is_categorical[attribute_i]
			self.raw_features.append(rf)

		print(self.raw_features)

		return self.raw_features

