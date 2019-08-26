from fastsklearnfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import numpy
from sklearn.pipeline import Pipeline

class ClassifierTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, classifier):
		self.classifier = classifier


	def fit(self, X, y=None):
		self.classifier.fit(X, y)
		return self

	def transform(self, X):
		'''
		model = self.pipeline.named_steps['c']
		one_class_index = -1
		for class_index in range(len(model.classes_)):
			if model.classes_[class_index] == 1:
				one_class_index = class_index
				break
		'''

		y_hat = self.classifier.predict_proba(X)[:, 0].reshape(-1, 1)
		return y_hat

