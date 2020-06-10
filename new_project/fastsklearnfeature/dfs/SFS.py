from fastsklearnfeature.dfs.BaseSelection import BaseSelection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_selection

class SFS(BaseSelection):
	def __init__(self):
		super(SFS, self).__init__(selection_function=forward_selection)
