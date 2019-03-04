import numpy as np
from autofeat import AutoFeatRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd

X = np.load('/tmp/X.npy')
X = X.astype(np.float64)
target = np.load('/tmp/y.npy')
target = target.astype(np.float64)



score=make_scorer(roc_auc_score, average='micro')
classifier = LogisticRegression()
parameters = {'penalty': ['l2'],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             'solver': ['lbfgs'],
             'class_weight': ['balanced'],
             'max_iter': [100000]
             }

clf = GridSearchCV(classifier, parameters, cv=10, scoring=score, iid=False, error_score='raise')
clf.fit(X, target)
probabilities = clf.predict_proba(X)

target = probabilities[:,1]

print(target)





#target = np.array(list(target))

print(type(X))
print(type(target))

print(X.shape)
print(target.shape)

'''
np.random.seed(15)
x1 = np.random.rand(1000)
x2 = np.random.randn(1000)
x3 = np.random.rand(1000)
target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 + np.log(x1))**3
X = np.vstack([x1, x2, x3]).T
'''

feateng_cols = ['age', 'sex', 'chest', 'resting_blood_pressure', 'serum_cholestoral', 'fasting_blood_sugar', 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak', 'slope', 'number_of_major_vessels', 'thal']


afreg = AutoFeatRegression(n_jobs=4, feateng_cols=feateng_cols)
df = afreg.fit_transform(pd.DataFrame(data=X, columns=feateng_cols), target)



