from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
import numpy as np
from sklearn.pipeline import Pipeline
from fastsklearnfeature.feature_selection.helper.ClassifierTransformer import ClassifierTransformer
from sklearn.pipeline import FeatureUnion

class AICEnsemble(BaseEstimator, ClassifierMixin):
	def __init__(self, candidateFeatures: List[CandidateFeature], classifier):
		self.candidateFeatures = candidateFeatures
		self.classifier = classifier

		self.ensemble_pipeline = FeatureUnion(transformer_list=[(str(c), self.generate_pipeline(c)) for c in candidateFeatures])

		# calculate weights
		self.AICc = np.array([np.min(c.runtime_properties['additional_metrics']['AICc_complexity']) for c in candidateFeatures])
		#self.AICc = [np.mean(c.runtime_properties['additional_metrics']['AICc_complexity']) for c in candidateFeatures]

		delta_i = self.AICc - np.min(self.AICc)
		summed = np.sum(np.array([np.exp(-delta_r / 2.0) for delta_r in delta_i]))
		self.weights = np.array([np.exp(-d_i / 2.0) / summed for d_i in delta_i])

		print(candidateFeatures)
		print(self.weights)



	def generate_pipeline(self, rep):
		best_hyperparameters = rep.runtime_properties['hyperparameters']

		all_keys = list(best_hyperparameters.keys())
		for k in all_keys:
			if 'classifier__' in k:
				best_hyperparameters[k[12:]] = best_hyperparameters.pop(k)

		my_pipeline = Pipeline([(str(rep) + '_f', rep.pipeline),
								(str(rep) + '_c', ClassifierTransformer(self.classifier(**best_hyperparameters)))
								])

		return my_pipeline

	def fit(self, X, y=None):
		self.ensemble_pipeline.fit(X, y)
		return self

	def predict_proba(self, X):
		ensemble_predictions = self.ensemble_pipeline.transform(X)

		print(ensemble_predictions)
		print(ensemble_predictions.shape)

		#weight these predictions
		weighted_predictions = np.multiply(ensemble_predictions, self.weights)

		averaged_predictions = np.sum(weighted_predictions, axis=1)

		averaged_predictions_proba = np.zeros((averaged_predictions.shape[0], 2))

		averaged_predictions_proba[:, 0] = averaged_predictions
		averaged_predictions_proba[:, 1] = 1.0 - averaged_predictions
		return averaged_predictions_proba

	def predict(self, X):
		return self.predict_proba(X)[:,0] < 0.5

