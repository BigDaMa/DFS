from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.notebook.api.SimulatedAnnealing import SimulatedAnnealing
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.notebook.api.TPEChi2 import TPEChi2
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.notebook.api.SFS import SFS
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.openml_data.notebook.api.DataLoader import DataLoader
import numpy as np

dl = DataLoader()
X_train, X_validation, X_test, y_train, y_validation, y_test, feature_names, sensitive_ids = dl.get_data(dataset='Adult')

sa_nr = SimulatedAnnealing()
sa_nr.query(X_train,
			X_validation,
			X_test,
			y_train,
			y_validation,
			y_test,
			classifier=LogisticRegression(class_weight='balanced'),
			min_accuracy=0.8,
			sensitive_ids=sensitive_ids,
			min_fairness=0.8,
			min_safety=0.0,
			min_privacy=None,
			max_complexity=10,
			max_search_time=np.inf,
			feature_names=feature_names
			)

sa_nr.get_progress()

sa_nr.get_test_radar_chart()