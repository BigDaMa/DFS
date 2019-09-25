# define a search space
from hyperopt import hp
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
import numpy as np
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import fmin, tpe, space_eval
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set


numeric_representations: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/features.p", "rb"))

filtered = numeric_representations
'''
filtered = []
for f in numeric_representations:
	if isinstance(f, RawFeature):
		filtered.append(f)
	else:
		if isinstance(f.transformation, OneHotTransformation):
			filtered.append(f)
'''

y_test = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/y_test.p", "rb"))


# define an objective function
def objective(args):
	features = []
	for f in range(len(numeric_representations)):
		features.append(args[str(numeric_representations[f])])
	score, test, pred_test = run_pipeline(features, c=args['C'])
	return {'loss': 1.0 - score, 'status': STATUS_OK, 'test' : test, 'pred_test': pred_test}







#hp.pchoice('fsd',[(p, True), (1-p, False)])


'''
def exp_normalize2(x):
	b = x.max()
	y = np.exp(x - b)
	return y

x = np.array([f.runtime_properties['score'] for f in numeric_representations])

probabilities = exp_normalize2(x) - 0.1

for f in range(len(numeric_representations)):
	space[str(numeric_representations[f])] = hp.pchoice(str(numeric_representations[f]),[(probabilities[f], True), (1-probabilities[f], False)])
'''




#hp.pchoice

# minimize the objective over the space

#trials: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new.p", "rb"))

trials = Trials()


space = {'C': hp.loguniform('C', np.log(1e-5), np.log(1e5))}
for f in range(len(numeric_representations)):
	space[str(numeric_representations[f])] = hp.choice(str(numeric_representations[f]), (True, False))

best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)


pred_best = trials.best_trial['result']['pred_test']
are_predictions_correct = np.equal(np.array(y_test.values), np.array(pred_best))


