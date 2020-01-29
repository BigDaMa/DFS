import numpy as np
from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier

from art.attacks import FastGradientMethod
from sklearn.model_selection import GridSearchCV
import copy

def robust_score(y_true, y_pred, eps=0.1, X=None, y=None, model=None, feature_selector=None, scorer=None):
	all_ids = range(X.shape[0])
	test_ids = y_true.index.values
	train_ids = list(set(all_ids)-set(test_ids))

	X_train = X[train_ids,:]
	y_train = y[train_ids]
	X_test = X[test_ids,:]
	y_test = y[test_ids]

	if type(feature_selector) != type(None):
		X_train = feature_selector.fit_transform(X_train)
		X_test = feature_selector.transform(X_test)


	#tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
	#cv = GridSearchCV(model, tuned_parameters)
	#cv.fit(X_train, y_train)
	#best_model = cv.best_estimator_
	best_model = copy.deepcopy(model)
	best_model.fit(X_train, y_train)

	classifier = SklearnClassifier(model=best_model)
	attack = FastGradientMethod(classifier, eps=eps, batch_size=1)

	X_test_adv = attack.generate(X_test)

	diff = scorer(best_model, X_test, y_test) - scorer(best_model, X_test_adv, y_test)
	return diff


def robust_score_test(eps=0.1, X_test=None, y_test=None, model=None, feature_selector=None, scorer=None):
	X_test_filtered = feature_selector.transform(X_test)


	best_model = copy.deepcopy(model)

	classifier = SklearnClassifier(model=best_model)
	attack = FastGradientMethod(classifier, eps=eps, batch_size=1)

	X_test_adv = attack.generate(X_test_filtered)

	score_original_test = scorer(best_model, X_test_filtered, y_test)
	score_corrupted_test = scorer(best_model, X_test_adv, y_test)

	diff = score_original_test - score_corrupted_test
	return diff



def unit_test_score(y_true, y_pred, unit_x=None, unit_y=None, X=None, y=None, pipeline=None):
	all_ids = range(X.shape[0])
	test_ids = y_true.index.values
	train_ids = list(set(all_ids)-set(test_ids))

	X_train = X[train_ids, :]
	y_train = y[train_ids]

	pipeline.fit(X_train, y_train)

	class_id = -1
	for c_i in range(len(pipeline.classes_)):
		if pipeline.classes_[c_i] == unit_y:
			class_id = c_i
			break
	y_pred = pipeline.predict_proba(np.array([unit_x]))[0, class_id]
	return y_pred