from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.pipeline import Pipeline
from fastsklearnfeature.transformations.sampling.PipelineTransformation import PipelineTransformation

from imblearn.pipeline import Pipeline as ImbalancePipeline
from imblearn.over_sampling import SMOTE

def generate_pipeline(feature: CandidateFeature, model):
	my_pipeline = Pipeline([('f', feature.pipeline),
							('c', model())
							])
	return my_pipeline

def generate_smote_pipeline(feature: CandidateFeature, model):
	my_pipeline = ImbalancePipeline([('f', PipelineTransformation(feature.pipeline)),
							('smote', SMOTE()),
							('c', model())
							])
	return my_pipeline