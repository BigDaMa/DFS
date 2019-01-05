from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV


X = np.random.randint(0.0, 1.0, size=(10,5))
y = [1,1,1,1,1,0,0,0,0,0]

parameters = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clf = GridSearchCV(LogisticRegression(), parameters, cv=5)
clf.fit(X, y)
score = clf.best_score_
print(score)

print(type(clf.scorer_))
