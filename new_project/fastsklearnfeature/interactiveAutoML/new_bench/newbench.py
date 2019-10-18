import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_oneway
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

from fastsklearnfeature.interactiveAutoML.feature_selection.L1Selection import L1Selection
from fastsklearnfeature.interactiveAutoML.feature_selection.MaskSelection import MaskSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.RedundancyRemoval import RedundancyRemoval
from fastsklearnfeature.interactiveAutoML.feature_selection.MajoritySelection import MajoritySelection

from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstruction import ComplexityDrivenFeatureConstruction
from fastsklearnfeature.reader.ScikitReader import ScikitReader
from fastsklearnfeature.transformations.MinusTransformation import MinusTransformation
from fastsklearnfeature.interactiveAutoML.feature_selection.ConstructionTransformation import ConstructionTransformer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle

'''
which_experiment = 'madelon'#'experiment3'#'experiment1'

numeric_representations: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/features.p", "rb"))

y_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_test.p", "rb")).values
#print(y_test)

my_names: List[CandidateFeature] = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/names.p", "rb"))
print(my_names)


X_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/X_train.p", "rb"))
X_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/X_test.p", "rb"))
y_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_train.p", "rb")).values
y_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_test.p", "rb")).values
'''


#Madelon:
X_train = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_train.data', delimiter=' ', header=None).values[:,0:500]
y_train = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_train.labels', delimiter=' ', header=None).values

X_test = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_valid.data', delimiter=' ', header=None).values[:,0:500]
y_test = pd.read_csv('/home/felix/Software/UCI-Madelon-Dataset/assets/madelon_valid.labels', delimiter=' ', header=None).values


'''
#ARCENE:
X_train = pd.read_csv('/home/felix/phd/feature_constraints/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000]
y_train = pd.read_csv('/home/felix/phd/feature_constraints/ARCENE/arcene_train.labels', delimiter=' ', header=None).values

X_test = pd.read_csv('/home/felix/phd/feature_constraints/ARCENE/arcene_valid.data', delimiter=' ', header=None).values[:,0:10000]
y_test = pd.read_csv('/home/felix/phd/feature_constraints/ARCENE/arcene_valid.labels', delimiter=' ', header=None).values
'''

'''
#credit data
data = pd.read_csv('/home/felix/phd/interactiveAutoML/dataset_31_credit-g.csv')
class_column_name = 'class'

y = data[class_column_name]
data_no_class = data[data.columns.difference([class_column_name])]

X = data_no_class.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42, stratify=y)
'''


parameter_grid = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'],
				  'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class': ['auto']}

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(X_train, y_train, feature_names=['V' + str(i) for i in range(X_train.shape[1])]),
                                                      score=auc, c_max=1, folds=2,
                                                      classifier=LogisticRegression,
                                                      grid_search_parameters=parameter_grid, n_jobs=4,
                                                      epsilon=0.0)

fe.run()



numeric_representations = []

feature_names = []
for r in fe.all_representations:
	if 'score' in r.runtime_properties:
		if not 'object' in str(r.properties['type']):
			if not isinstance(r.transformation, MinusTransformation):
				print(str(r) + ':' + str(r.properties['type']) + ' : ' + str(r.runtime_properties['score']))
				numeric_representations.append(r)
				feature_names.append(str(r))




ground_truth = [28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378, 433, 442, 451, 453, 455, 472, 475, 493]


print(len(ground_truth))

mask = np.zeros(len(numeric_representations), dtype=bool)
for i in range(len(numeric_representations)):
	for g in ground_truth:
		if str(numeric_representations[i]) == 'V' + str(g):
			mask[i] = True
			break

print(np.sum(mask))

all_features = CandidateFeature(IdentityTransformation(-1), numeric_representations)
all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])

#foreigner = np.array(X_train[:,7])
#gender = np.array(['female' in personal_status for personal_status in X_train[:,15]])

my_pipeline = Pipeline([('features', all_standardized.pipeline),
						#('selection', L1Selection()),
						#('selection', SelectKBest(score_func=f_classif)),
						#('selection', SelectKBest(score_func=mutual_info_classif,k=20)),
						#('selection', SelectKBest(score_func=chi2,k=20)),
						#('selection', SelectKBest(score_func=f_oneway,k=20)),
						('selection', RFE(LogisticRegression(penalty='l1', C=0.1))),
						#('selection', SelectFromModel(LogisticRegression(penalty='l1', C=0.0375))),
						#('selection', SelectFromModel(DecisionTreeClassifier())),
						#('selection', RedundancyRemoval()), #takes really long
						#('selection', MajoritySelection([SelectKBest(score_func=f_classif, k=20), SelectKBest(score_func=mutual_info_classif, k=20), SelectFromModel(DecisionTreeClassifier())])),
						#('selection', MaskSelection(mask)),
						#('new_construction', ConstructionTransformer(c_max=3, scoring=auc, n_jobs=4, model=LogisticRegression(), cv=10)), #helps to uncover non-linear relationships
						#('selection', SelectFromModel(DecisionTreeClassifier())),
						#('pca', PCA(n_components=10)), #no improvement
 						('model', LogisticRegression())
                        #('model', KNeighborsClassifier())
					   ])

'''
parameter_grid = {'model__penalty': ['l2'], 'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'model__solver': ['lbfgs'],
						  'model__class_weight': ['balanced'], 'model__max_iter': [10000], 'model__multi_class': ['auto']}
'''
parameter_grid = {
				  #'selection__k': range(1, X_train.shape[1]),
				  'selection__n_features_to_select': range(X_train.shape[1]-1, 450, -1),

				  'model__penalty': ['l2'],
				  'model__C': [1],
				  'model__solver': ['lbfgs'],
				  'model__class_weight': ['balanced'],
				  'model__max_iter': [10000],
				  'model__multi_class': ['auto']}

#parameter_grid = {'model__n_neighbors': [3]}

scoring = {'auc': make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)}


kfolds = StratifiedKFold(10, shuffle=True, random_state=42)
gridcv = GridSearchCV(my_pipeline, parameter_grid, cv=kfolds.split(X_train, y_train), scoring=scoring, n_jobs=4, refit='auc')
gridcv.fit(X_train, pd.DataFrame(y_train))

feature_precision = precision_score(mask, gridcv.best_estimator_.named_steps['selection']._get_support_mask())
feature_recall = recall_score(mask, gridcv.best_estimator_.named_steps['selection']._get_support_mask())

print("Feature precision: " + str(feature_precision) + ' Feature recall: ' + str(feature_recall))


test_score = gridcv.score(X_test, pd.DataFrame(y_test))

print('cv score: ' + str(gridcv.best_score_))

pickle.dump(gridcv.cv_results_, open("/home/felix/phd/feature_constraints/bench/Kbest_grid.p", "wb"))






'''

parameter_grid = {'model__penalty': 'l2', 'model__C': 1, 'model__solver': 'lbfgs',
						  'model__class_weight': 'balanced', 'model__max_iter': 10000, 'model__multi_class': 'auto'}

my_pipeline.set_params(**parameter_grid)
my_pipeline.fit(X_train, pd.DataFrame(y_train))


test_score = auc(my_pipeline, X_test, pd.DataFrame(y_test))
print('test score: ' + str(test_score))
'''
