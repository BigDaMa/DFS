import numpy as np
from sklearn.linear_model import LogisticRegression
import diffprivlib.models as models
import time
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.multiprocessing_global as mp_global
import copy
import pandas as pd
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.utils.gridsearch import run_grid_search
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.robust_measure import robust_score_test
from fastsklearnfeature.interactiveAutoML.fair_measure import true_positive_rate_score
from sklearn.metrics import f1_score

def load_pickle(fname):
	data = []
	with open(fname, "rb") as f:
		while True:
			try:
				data.append(pickle.load(f))
			except EOFError:
				break
	return data


def is_successfull_validation_and_test(exp_results):
	return len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1]['success_test'] == True #also on test satisfied

def is_successfull_validation(exp_results):
	return len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[-1]  # constraints were satisfied on validation set

def my_function(config_id):
	conf = mp_global.configurations[config_id]
	result = conf['main_strategy'](mp_global.X_train,
								   mp_global.X_validation,
								   mp_global.X_train_val,
								   mp_global.X_test,
								   mp_global.y_train,
								   mp_global.y_validation,
								   mp_global.y_train_val,
								   mp_global.y_test,
								   mp_global.names,
								   mp_global.sensitive_ids,
								   ranking_functions=conf['ranking_functions'],
								   clf=mp_global.clf,
								   min_accuracy=mp_global.min_accuracy,
								   min_fairness=mp_global.min_fairness,
								   min_robustness=mp_global.min_robustness,
								   max_number_features=mp_global.max_number_features,
								   max_search_time=mp_global.max_search_time,
								   log_file=mp_global.log_file)
	return result


class BaseSelection(object):
	def __init__(self, selection_function, ranking_functions=[]):
		self.selection_function = selection_function
		self.ranking_functions = ranking_functions
		self.stored_results_file = None
		self.feature_names = None

	def query(self,
			  X_train,
			  X_validation,
			  X_test,
			  y_train,
			  y_validation,
			  y_test,
			  classifier=LogisticRegression(class_weight='balanced'),
			  min_accuracy=0.5,
			  sensitive_ids=None,
			  min_fairness=0.0,
			  min_safety=0.0,
			  min_privacy=None,
			  max_complexity=1.0,
			  max_search_time=np.inf,
			  feature_names=None
			  ):

		if isinstance(max_complexity, int):
			max_complexity = max_complexity / float(X_train.shape[1])

		X_train_val = np.vstack((X_train, X_validation))
		y_train_val = np.append(y_train, y_validation)

		self.feature_names = feature_names

		if type(min_privacy) != type(None):
			classifier = models.LogisticRegression(epsilon=min_privacy, class_weight='balanced')

		self.stored_results_file = '/tmp/experiment' + str(time.time()) + '.pickle'

		mp_global.X_train = X_train
		mp_global.X_validation = X_validation
		mp_global.X_train_val = X_train_val
		mp_global.X_test = X_test
		mp_global.y_train = y_train
		mp_global.y_validation = y_validation
		mp_global.y_train_val = y_train_val
		mp_global.y_test = y_test
		mp_global.names = feature_names
		mp_global.sensitive_ids = sensitive_ids

		mp_global.min_accuracy = min_accuracy
		mp_global.min_fairness = min_fairness
		mp_global.min_robustness = min_safety
		mp_global.max_number_features = max_complexity
		mp_global.max_search_time = max_search_time
		mp_global.clf = classifier
		mp_global.log_file = self.stored_results_file

		configuration = {}
		configuration['ranking_functions'] = copy.deepcopy(self.ranking_functions)
		configuration['run_id'] = 0
		configuration['main_strategy'] = copy.deepcopy(self.selection_function)

		mp_global.configurations = [configuration]

		with ProcessPool(max_workers=1) as pool:
			future = pool.map(my_function, range(len(mp_global.configurations)), timeout=max_search_time)

			iterator = future.result()
			while True:
				try:
					result = next(iterator)
				except StopIteration:
					break
				except TimeoutError as error:
					print("function took longer than %d seconds" % error.args[1])
				except ProcessExpired as error:
					print("%s. Exit code: %d" % (error, error.exitcode))
				except Exception as error:
					print("function raised %s" % error)
					#print(error.traceback)  # Python's traceback of remote process


		return self.get_satisfying_features()

	def get_satisfying_features(self):
		exp_results = []
		try:
			exp_results = load_pickle(self.stored_results_file)
		except:
			pass
		if is_successfull_validation_and_test(exp_results):
			mask = exp_results[-1]['selected_features']._get_support_mask()
			indices = np.nonzero(mask)
			return np.array(self.feature_names)[indices]
		else:
			return np.array([])


	def get_trained_model(self):
		return None



	def read_file(self):
		exp_results = []
		try:
			exp_results = load_pickle(self.stored_results_file)
		except:
			pass
		return exp_results

	def get_progress(self):
		exp_results = self.read_file()

		times = []
		loss = []

		for e in exp_results:
			loss.append(e['loss'])
			times.append(e['time'])

		plt.plot(times, loss)
		plt.xlabel('Search Time (Seconds)')
		plt.ylabel('Distance to specified constraints')
		plt.title('Search Progress')
		plt.show()





	def transfer_to_other_model(self, model,
			  X_train,
			  X_validation,
			  X_test,
			  y_train,
			  y_validation,
			  y_test,
			  sensitive_ids=None,
			  hyperparameters=None):
		exp_results = self.read_file()

		X_train_val = np.vstack((X_train, X_validation))
		y_train_val = np.append(y_train, y_validation)

		min_loss = np.inf
		best_run = None
		for min_r in range(len(exp_results)):
			if 'loss' in exp_results[min_r] and exp_results[min_r]['loss'] < min_loss:
				min_loss = exp_results[min_r]['loss']
				best_run = min_r

		feature_selection = exp_results[best_run]['selected_features']

		start_time = time.time()

		pipeline = Pipeline([
			('selection', feature_selection),
			('clf', model)
		])

		accuracy_scorer = make_scorer(f1_score)
		grid_result = run_grid_search(pipeline, X_train, y_train, X_validation, y_validation,
									  accuracy_scorer, sensitive_ids,
									  1.0, 1.0, 1.0, 0.0,
									  model_hyperparameters=hyperparameters,
									  start_time=start_time)

		model = grid_result['model']
		model.fit(X_train_val, pd.DataFrame(y_train_val))

		test_acc = accuracy_scorer(model, X_test, pd.DataFrame(y_test))

		fair_test = None
		if type(sensitive_ids) != type(None):
			fair_test = make_scorer(true_positive_rate_score, greater_is_better=True,
									sensitive_data=X_test[:, sensitive_ids[0]])

		test_fair = 0.0
		if type(sensitive_ids) != type(None):
			test_fair = 1.0 - fair_test(model, X_test, pd.DataFrame(y_test))
		test_robust = 1.0 - robust_score_test(eps=0.1, X_test=X_test, y_test=y_test,
											  model=model.named_steps['clf'],
											  feature_selector=model.named_steps['selection'],
											  scorer=accuracy_scorer)

		test_number_features = exp_results[best_run]['cv_number_features']

		categories = ['Accuracy', 'Fairness', 'Simplicity', 'Safety']



		fig = go.Figure()

		fig.add_trace(go.Scatterpolar(
			r=[test_acc, test_fair, 1.0 - test_number_features, test_robust],
			theta=categories,
			fill='toself',
			name='Best Result on Test'
		))

		fig.update_layout(
			title='Best Result on Test',
			polar=dict(
				radialaxis=dict(
					visible=True,
					range=[0, 1]
				)),
			showlegend=False
		)

		return fig



	def get_test_radar_chart(self):
		exp_results = self.read_file()

		min_loss = np.inf
		best_run = None
		for min_r in range(len(exp_results)):
			if 'loss' in exp_results[min_r] and exp_results[min_r]['loss'] < min_loss:
				min_loss = exp_results[min_r]['loss']
				best_run = min_r

		test_fair = exp_results[best_run]['test_fair']
		test_acc = exp_results[best_run]['test_acc']
		test_robust = exp_results[best_run]['test_robust']
		test_number_features = exp_results[best_run]['cv_number_features']

		categories = ['Accuracy', 'Fairness', 'Simplicity', 'Safety']



		fig = go.Figure()

		fig.add_trace(go.Scatterpolar(
			r=[test_acc, test_fair, 1.0 - test_number_features, test_robust],
			theta=categories,
			fill='toself',
			name='Best Result on Test'
		))

		fig.update_layout(
			title='Best Result on Test',
			polar=dict(
				radialaxis=dict(
					visible=True,
					range=[0, 1]
				)),
			showlegend=False
		)

		return fig

