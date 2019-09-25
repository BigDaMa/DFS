import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import roc_auc_score
import eli5
from yellowbrick.features import rank2d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.feature_selection.ComplexityDrivenFeatureConstruction import ComplexityDrivenFeatureConstruction
from fastsklearnfeature.reader.ScikitReader import ScikitReader
from fastsklearnfeature.transformations.MinusTransformation import MinusTransformation
import pickle

data = pd.read_csv('/home/felix/phd/interactiveAutoML/dataset_31_credit-g.csv')

y = data['class']

data_no_class = data[data.columns.difference(['class'])]

X = data[data.columns.difference(['class'])].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42, stratify=y)

model = LogisticRegression
parameter_grid = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs'],
				  'class_weight': ['balanced'], 'max_iter': [10000], 'multi_class': ['auto']}

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

fe = ComplexityDrivenFeatureConstruction(None, reader=ScikitReader(X_train, y_train,
                                                                                feature_names=data[data.columns.difference(['class'])].columns),
                                                      score=auc, c_max=3, folds=10,
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

print(numeric_representations)
print(len(numeric_representations))

#store features
#store data

#then, we can code a program that delivers everything we need

pickle.dump(numeric_representations, open("/home/felix/phd/feature_constraints/experiment3/features.p", "wb"))

#X_train, X_test, y_train, y_test

pickle.dump(data[data.columns.difference(['class'])].columns, open("/home/felix/phd/feature_constraints/experiment3/names.p", "wb"))

pickle.dump(X_train, open("/home/felix/phd/feature_constraints/experiment3/X_train.p", "wb"))
pickle.dump(X_test, open("/home/felix/phd/feature_constraints/experiment3/X_test.p", "wb"))
pickle.dump(y_train, open("/home/felix/phd/feature_constraints/experiment3/y_train.p", "wb"))
pickle.dump(y_test, open("/home/felix/phd/feature_constraints/experiment3/y_test.p", "wb"))
