import copy
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder


def run_experiments_for_strategy(X_train, y_train, data_name, my_search_strategy = run_hyperopt_search_kbest_info, max_time =20 * 60):

	name = my_search_strategy.__name__
	xshape = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(X_train).shape[1]

	# generate grid
	complexity_grid = np.arange(1, xshape + 1)
	max_acc = 1.0
	#accuracy_grid = np.arange(0.0, max_acc, max_acc / len(complexity_grid))
	accuracy_grid = np.arange(0.0, max_acc, max_acc / 100.0)


	#print(complexity_grid)
	#print(accuracy_grid)

	grid = list(itertools.product(complexity_grid, accuracy_grid))

	#print(len(grid))

	meta_X_data = np.matrix(grid)


	#run 10 random combinations

	random_combinations = 10
	ids = []
	#ids = np.random.choice(len(grid), size=random_combinations, replace=False, p=None)

	for i in range(0, len(accuracy_grid), int(len(accuracy_grid)/float(random_combinations))):
		ids.append(len(grid) - i - 1)



	kfold = StratifiedKFold(n_splits=10, shuffle=False)
	scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

	meta_X_train = np.zeros((random_combinations, 2))
	runtimes = []
	success_check = []


	for rounds in range(20):
		print("number of ids: " + str(len(ids)))
		for i in range(len(ids)):
			complexity = meta_X_data[ids[i], 0]
			accuracy = meta_X_data[ids[i], 1]
			print("min acc: " + str(accuracy))
			#try:
			runtime = my_search_strategy(X_train, y_train,
											model=DecisionTreeClassifier(),
											kfold=copy.deepcopy(kfold),
											scoring=scoring,
											max_complexity=int(complexity),
											min_accuracy=accuracy,
											fit_time_out=max_time)
			success_check.append(True)
			#except:
			#	runtime = max_time
			#	success_check.append(False)
			#	print("did not find a solution")
			print(runtime)

			if rounds==0:
				meta_X_train[i] = meta_X_data[ids[i]]
			else:
				meta_X_train = np.vstack([meta_X_train, meta_X_data[ids[i]]])
			runtimes.append(runtime)

		al_model = RandomForestRegressor(n_estimators=10)
		al_model.fit(meta_X_train, runtimes)

		pfile = open("/tmp/model" + str(meta_X_train.shape[0]) + "_" + name + '_data_' + data_name +".p", "wb")
		pickle.dump(al_model, pfile)
		pfile.flush()
		pfile.close()

		print(runtimes)

		# calculate uncertainty of predictions for sampled pairs
		predictions = []
		for tree in range(al_model.n_estimators):
			predictions.append(al_model.estimators_[tree].predict(meta_X_data))

		print(predictions)

		uncertainty = np.matrix(np.std(np.matrix(predictions).transpose(), axis=1)).A1

		print('mean uncertainty: ' + str(np.average(uncertainty)))

		uncertainty_sorted_ids = np.argsort(uncertainty * -1)
		ids = [uncertainty_sorted_ids[0]]

		#predict search failure
		if len(np.unique(np.array(success_check))) == 2:
			al_success_model = RandomForestClassifier(n_estimators=10)
			al_success_model.fit(meta_X_train, success_check)

			pfile = open("/tmp/success_model" + str(meta_X_train.shape[0]) + "_" + name + '_data_' + data_name +".p", "wb")
			pickle.dump(al_success_model, pfile)
			pfile.flush()
			pfile.close()

			# calculate uncertainty of predictions for sampled pairs
			predictions = []
			for tree in range(al_success_model.n_estimators):
				predictions.append(al_success_model.estimators_[tree].predict_proba(meta_X_data)[:, 0])

			print(predictions)

			uncertainty = np.matrix(np.std(np.matrix(predictions).transpose(), axis=1)).A1

			print('mean uncertainty: ' + str(np.average(uncertainty)))

			uncertainty_sorted_ids = np.argsort(uncertainty * -1)
			ids.append(uncertainty_sorted_ids[0])




		'''
		runtime_predictions = al_model.predict(meta_X_data)

		df = pd.DataFrame.from_dict(np.array([meta_X_data[:, 0].A1, meta_X_data[:, 1].A1, runtime_predictions]).T)
		df.columns = ['Max Complexity', 'Min Accuracy', 'Estimated Runtime']
		pivotted = df.pivot('Max Complexity', 'Min Accuracy', 'Estimated Runtime')
		sns_plot = sns.heatmap(pivotted, cmap='RdBu')
		fig = sns_plot.get_figure()
		fig.savefig("/tmp/output" + str(meta_X_train.shape[0]) + "_" + name + '_data_' + data_name +".png", bbox_inches='tight')
		plt.clf()
		'''

'''
X_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.data', delimiter=' ', header=None).values[:,0:10000][0:100,:]
y_train = pd.read_csv(Config.get('data_path') + '/ARCENE/arcene_train.labels', delimiter=' ', header=None).values[0:100]
run_experiments_for_strategy(X_train, y_train, 'arcene_sample', run_hyperopt_search, max_time = 20 * 60)
'''
