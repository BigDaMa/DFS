from fastsklearnfeature.dfs.BaseSelection import BaseSelection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import TPE


class TPE_NR(BaseSelection):
	def __init__(self):
		super(TPE_NR, self).__init__(selection_function=TPE)
