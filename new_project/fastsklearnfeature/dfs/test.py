from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.dfs.SimulatedAnnealing import SimulatedAnnealing
from fastsklearnfeature.dfs.DataLoader import DataLoader
import numpy as np

dl = DataLoader()
X_train, X_validation, X_test, y_train, y_validation, y_test, feature_names, sensitive_ids = dl.get_data(dataset='German Credit')

sa_nr = SimulatedAnnealing()
sa_nr.query(X_train,
			X_validation,
			X_test,
			y_train,
			y_validation,
			y_test,
			classifier=LogisticRegression(class_weight='balanced'),
			min_accuracy=np.inf,
			sensitive_ids=sensitive_ids,
			min_fairness=0.5,
			min_safety=0.0,
			min_privacy=None,
			max_complexity=10,
			max_search_time=60,
			feature_names=feature_names
			)

sa_nr.get_progress()

sa_nr.get_test_radar_chart()