import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

X_data = np.random.randint(0, 10, (100, 5))
y_data = np.random.randint(0, 1, 100)
groups = np.random.randint(0, 5, 100)

def test_score(y_true, y_pred):
	indices = y_true.index.values
	print(indices)
	return 0


y_data = pd.DataFrame(y_data)
y_data = y_data.iloc[50:100]

my_score = make_scorer(test_score, greater_is_better=False)
scores = cross_val_score(RandomForestClassifier(), (X_data[50:100,:])[10:20,:], y_data.iloc[10:20], cv=5, scoring=my_score)