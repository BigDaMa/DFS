import pandas as pd
from typing import List
from fastsklearnfeature.candidates.RawFeature import RawFeature

class ScikitReader:
	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		self.raw_features: List[RawFeature] = []

	def read(self) -> List[RawFeature]:

		self.dataframe = pd.DataFrame(data=self.X_train)

		self.splitted_values = {}
		self.splitted_target = {}

		self.splitted_target['train'] = self.y_train
		self.splitted_target['test'] = []
		self.splitted_values['train'] = self.X_train
		self.splitted_values['test'] = []

		for attribute_i in range(self.dataframe.shape[1]):
			rf = RawFeature('Feature' + str(self.dataframe.columns[attribute_i]), attribute_i, {})
			rf.derive_properties(self.dataframe[self.dataframe.columns[attribute_i]].values)
			self.raw_features.append(rf)

		return self.raw_features

