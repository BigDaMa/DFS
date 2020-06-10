from fastsklearnfeature.dfs.BaseSelection import BaseSelection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo

class TPEChi2(BaseSelection):
	def __init__(self):
		super(TPEChi2, self).__init__(selection_function=weighted_ranking, ranking_functions=[chi2_score_wo])
