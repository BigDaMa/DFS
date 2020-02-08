import numpy as np
from sklearn.metrics import confusion_matrix


def time_score(y_true, y_pred, logs=None, max_time=20*60):
	indices = y_true.index.values

	print(y_pred)

	loss = 0
	for y_i in range(len(y_pred)):
		current_strategy = y_pred[y_i]
		current_id = indices[y_i]
		if current_strategy in logs['times_value'][current_id] and len(logs['times_value'][current_id][current_strategy]) >= 1:
			runtime_that_results_from_prediction = min(logs['times_value'][current_id][current_strategy])
		else:
			runtime_that_results_from_prediction = max_time
		best_strategy = logs['best_strategy'][current_id]
		best_runtime = 0
		if best_strategy != 0:
			if len(logs['times_value'][current_id][best_strategy]) == 0:
				continue
			else:
				best_runtime = min(logs['times_value'][current_id][best_strategy])

		if best_strategy == 0:
			if current_strategy != 0:
				#loss += max_time ** 2
				loss += 0
			else:
				loss += 0
		else:
			print("best runtime: " + str(best_runtime) + ' predicted: ' + str(runtime_that_results_from_prediction))

			if current_strategy != 0:
				loss += (runtime_that_results_from_prediction - best_runtime) #** 2
	return loss