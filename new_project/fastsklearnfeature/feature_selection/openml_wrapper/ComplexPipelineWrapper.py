from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class ComplexPipelineWrapper(BaseEstimator):

	def __init__(self, my_pipeline: Pipeline):
		self.my_pipeline = my_pipeline


	def fit(self, X, y=None, **fit_params):
		self.my_pipeline.fit(X, y, **fit_params)
		self.classes_ = self.my_pipeline.named_steps['classifier'].classes_
		return self


	def predict(self, X, **predict_params):
		return self.my_pipeline.predict(X, **predict_params)






