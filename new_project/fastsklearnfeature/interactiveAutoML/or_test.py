from __future__ import print_function
from ortools.linear_solver import pywraplp
import numpy as np
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
from ortools.sat.python import cp_model

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
def objective(features) -> float:
    score, test, pred_test = run_pipeline(features, c=1.0)
    print(score)
    return score


# Create the mip solver with the CBC backend.
model = cp_model.CpModel()


feature_bools = []
for i in range(len(numeric_representations)):
    #feature_bools.append(solver.BoolVar('x' + str(i)))
    feature_bools.append(model.NewBoolVar('x' + str(i)))


#print('Number of variables =', solver.NumVariables())

# x + 7 * y <= 17.5.
#solver.Add(np.sum([x, 7 * y]) <= 17.5)

# x <= 3.5.
model.Add(bool(objective(feature_bools) > 0.79))





solver = cp_model.CpSolver()
status = solver.Solve(model)


print(status)
print(solver.ObjectiveValue())