from hyperopt import hp

constraint_space = {
	'min_accuracy': 		hp.uniform('val', 0.5, 1),
    'max_search_time': 		hp.uniform('val', 10, 3 * 60 * 60), # in seconds
	'max_feature_fraction': hp.choice('?', [1, hp.uniform('val', 0, 1)]),
	'min_fairness': 		hp.choice('?', [0, hp.uniform('val', 0, 1)]),
	'min_robustness': 		hp.choice('?', [0, hp.uniform('val', 0, 1)]),
    'privacy epsilon':	    hp.choice('?', [None, hp.lognormal('val', 0, 1)])
}

import hyperopt.pyll.stochastic
print(hyperopt.pyll.stochastic.sample(constraint_space))

'''
space = {'test': hp.lognormal('privacy_specified', 0, 1)}

for i in range(1000):
	print(hyperopt.pyll.stochastic.sample(space))
'''