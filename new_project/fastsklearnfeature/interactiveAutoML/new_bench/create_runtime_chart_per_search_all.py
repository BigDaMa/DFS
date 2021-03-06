import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder


def run_experiments_for_strategy(X_train, y_train, data_name, my_search_strategy = None, max_time =20 * 60, one_hot=False):

	name = my_search_strategy.__name__

	xshape = X_train.shape[1]
	if one_hot:
		xshape = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(X_train).shape[1]

	# generate grid
	complexity_grid = np.arange(1, xshape + 1)
	max_acc = 1.0
	#accuracy_grid = np.arange(0.0, max_acc, max_acc / len(complexity_grid))
	accuracy_grid = np.arange(0.0, max_acc, max_acc / 100.0)


	kfold = StratifiedKFold(n_splits=10, shuffle=False)
	scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

	runtime_dict = {}
	success_dict = {}

	for accuracy in accuracy_grid:
		for complexity in complexity_grid:
			print("min acc: " + str(accuracy) + ' complexity: ' + str(complexity))
			try:
				runtime = my_search_strategy(X_train, y_train,
												model=DecisionTreeClassifier(),
												kfold=copy.deepcopy(kfold),
												scoring=scoring,
												max_complexity=int(complexity),
												min_accuracy=accuracy,
												fit_time_out=max_time,
												one_hot=one_hot
											 )
				success_dict[(accuracy, complexity)] = True
				runtime_dict[(accuracy, complexity)] = runtime
			except Exception as e:
				print(e)
				success_dict[(accuracy, complexity)] = False
				runtime_dict[(accuracy, complexity)] = max_time
				print("did not find a solution")

			pfile = open("/tmp/actual_results_" + name + '_data_' + data_name +".p", "wb")
			pickle.dump(runtime_dict, pfile)
			pfile.flush()
			pfile.close()

			pfile = open("/tmp/success_actual_results_" + name + '_data_' + data_name + ".p", "wb")
			pickle.dump(success_dict, pfile)
			pfile.flush()
			pfile.close()



