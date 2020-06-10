from fastsklearnfeature.dfs.BaseSelection import BaseSelection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import simulated_annealing


class SimulatedAnnealing(BaseSelection):
	def __init__(self):
		super(SimulatedAnnealing, self).__init__(selection_function=simulated_annealing)
