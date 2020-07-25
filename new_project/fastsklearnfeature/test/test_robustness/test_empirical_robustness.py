from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from art.metrics import empirical_robustness
from sklearn.model_selection import train_test_split
from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import BoundaryAttack
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy.linalg as la

from art.classifiers.scikitlearn import SklearnClassifier

import art

print(art.__version__)

diabetes = load_breast_cancer()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)


print(X_test.shape)

print("trained")

classifier = SklearnClassifier(model=model)
attack = HopSkipJump(classifier=classifier, max_iter=1, max_eval=10, init_eval=10, init_size=1)
X_test_attacked = attack.generate(X_test, y_test)


robustness = empirical_robustness(classifier, X_test, 'hsj', attack_params= {'max_iter': 1, 'max_eval': 10, 'init_eval': 10, 'init_size': 1})
print('Robustness: ' + str(robustness))

print("generated")

y_test_attacked = model.predict(X_test_attacked)
y_test_pred = model.predict(X_test)

print(f1_score(y_true=y_test, y_pred=y_test_pred))
print(f1_score(y_true=y_test, y_pred=y_test_attacked))

print('Robustness: ' + str(1.0 - (f1_score(y_true=y_test, y_pred=y_test_pred) - f1_score(y_true=y_test, y_pred=y_test_attacked))))

