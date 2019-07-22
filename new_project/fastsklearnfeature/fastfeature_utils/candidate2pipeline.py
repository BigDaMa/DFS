from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.pipeline import Pipeline

def generate_pipeline(feature: CandidateFeature, model):
	my_pipeline = Pipeline([('f', feature.pipeline),
							('c', model())
							])
	return my_pipeline