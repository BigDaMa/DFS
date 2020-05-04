from hyperopt import hp
constraint_space = {
'min_accuracy':
	hp.uniform('val', 0.5, 1),
'max_time':
	hp.uniform('val', 10, 3 * 60 * 60),
'max_features':
	hp.choice('?', [1, hp.uniform('val', 0, 1)]),
'min_fairness':
	hp.choice('?', [0, hp.uniform('val', 0, 1)]),
'min_safety':
	hp.choice('?', [0, hp.uniform('val', 0, 1)]),
'privacy_$\varepsilon$':
	hp.choice('?', [None, hp.lognormal('val', 0, 1)])
}

print(constraint_space)