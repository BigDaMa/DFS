import numpy as np
from art.classifiers import SklearnClassifier
from art.attacks import FastGradientMethod
from sklearn.model_selection import train_test_split

def robustness_score(X, y=None, model=None, scorer=None):
	X_train_rob, X_test_rob, y_train_rob, y_test_rob = train_test_split(X, y, test_size=0.5, random_state=42)

	robustness_ranking = np.zeros(X_train_rob.shape[1])

	for feature_i in range(X_train_rob.shape[1]):
		feature_ids = list(range(X_train_rob.shape[1]))
		del feature_ids[feature_i]

		model.fit(X_train_rob[:, feature_ids], y_train_rob)

		classifier = SklearnClassifier(model=model)
		attack = FastGradientMethod(classifier, eps=0.1, batch_size=1)

		X_test_adv = attack.generate(X_test_rob[:, feature_ids])

		diff = scorer(model, X_test_rob[:, feature_ids], y_test_rob) - scorer(model, X_test_adv, y_test_rob)
		robustness_ranking[feature_i] = diff
	robustness_ranking *= -1

	return robustness_ranking