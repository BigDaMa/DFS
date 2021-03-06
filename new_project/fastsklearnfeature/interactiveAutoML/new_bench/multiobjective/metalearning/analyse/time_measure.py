import numpy as np
from sklearn.metrics import confusion_matrix


def time_score2(y_true, y_pred, logs=None, squared=False, number_strategiesplus1=9):
	indices = y_true.index.values

	loss = 0
	for y_i in range(len(y_pred)):
		current_strategy = y_pred[y_i]
		current_id = indices[y_i]
		if current_strategy in logs['success_value'][current_id] and len(
				logs['success_value'][current_id][current_strategy]) > 0 and \
				logs['success_value'][current_id][current_strategy][0] == True:

			runtime_that_results_from_prediction = min(logs['times_value'][current_id][current_strategy])
		else:
			runtime_that_results_from_prediction = logs['features'][current_id][6]

		best_runtime = np.inf
		for s in range(1, number_strategiesplus1):
			if s in logs['success_value'][current_id] and len(
				logs['success_value'][current_id][s]) > 0 and \
				logs['success_value'][current_id][s][0] == True:
				runtime = min(logs['times_value'][current_id][s])
				if runtime < best_runtime:
					best_runtime = runtime

		if squared:
			loss += (runtime_that_results_from_prediction - best_runtime) ** 2
		else:
			loss += (runtime_that_results_from_prediction - best_runtime)
	return loss / float(len(y_pred))


def get_recall(y_true, y_pred, logs=None):
	indices = y_true.index.values

	loss = 0
	for y_i in range(len(y_pred)):
		current_strategy = y_pred[y_i]
		current_id = indices[y_i]

		if current_strategy in logs['success_value'][current_id] and len(logs['success_value'][current_id][current_strategy]) > 0 and logs['success_value'][current_id][current_strategy][0] == True:
			loss += 1
	return loss / float(len(y_pred))

def get_avg_runtime(y_true, y_pred, logs=None):
	indices = y_true.index.values

	loss = 0
	for y_i in range(len(y_pred)):
		current_strategy = y_pred[y_i]
		current_id = indices[y_i]

		if current_strategy in logs['success_value'][current_id] and len(
				logs['success_value'][current_id][current_strategy]) > 0 and \
				logs['success_value'][current_id][current_strategy][0] == True:
			loss += min(logs['times_value'][current_id][current_strategy])
		else:
			loss += logs['features'][current_id][6]
	return loss / float(len(y_pred))


def get_optimum_avg_runtime(y_true, y_pred, logs=None, number_strategiesplus1=9):
	indices = y_true.index.values

	loss = 0
	for y_i in range(len(y_pred)):
		current_id = indices[y_i]
		best_runtime = logs['features'][current_id][6]
		for s in range(1, number_strategiesplus1):
			if s in logs['success_value'][current_id] and len(logs['success_value'][current_id][s]) > 0 and logs['success_value'][current_id][s][0] == True:
				runtime = min(logs['times_value'][current_id][s])
				if runtime < best_runtime:
					best_runtime = runtime
		loss += best_runtime


	return loss / float(len(y_pred))