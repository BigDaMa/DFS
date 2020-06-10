from fastsklearnfeature.dfs.BaseSelection import BaseSelection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_floating_selection


class ForwardSelection(BaseSelection):
	def __init__(self, floating=False):
		if floating:
			super(ForwardSelection, self).__init__(selection_function=forward_floating_selection)
		else:
			super(ForwardSelection, self).__init__(selection_function=forward_selection)
