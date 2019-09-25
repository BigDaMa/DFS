from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np


which_experiment = 'experiment3'#'experiment1'

def run_pipeline(which_features_to_use, c=None, runs=1):

	model = LogisticRegression

	if type(c) == type(None):
		c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	else:
		c = [c]

	parameter_grid = {'c__penalty': ['l2'], 'c__C': c, 'c__solver': ['lbfgs'],
					  'c__class_weight': ['balanced'], 'c__max_iter': [10000], 'c__multi_class': ['auto']}

	auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

	numeric_representations = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/features.p", "rb"))

	#print(len(numeric_representations))

	#X_train, X_test, y_train, y_test
	X_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/X_train.p", "rb"))
	X_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/X_test.p", "rb"))
	y_train = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_train.p", "rb"))
	y_test = pickle.load(open("/home/felix/phd/feature_constraints/" + str(which_experiment) + "/y_test.p", "rb"))



	#generate pipeline
	all_selected_features = []
	for i in range(len(which_features_to_use)):
		if which_features_to_use[i]:
			all_selected_features.append(numeric_representations[i])

	all_features = CandidateFeature(IdentityTransformation(-1), all_selected_features)
	all_standardized = CandidateFeature(MinMaxScalingTransformation(), [all_features])

	my_pipeline = Pipeline([('f', all_standardized.pipeline),
							('c', model())
							])

	cv_scores = []
	test_scores = []
	pred_test = None

	if runs > 1:
		for r in range(runs):
			kfolds = StratifiedKFold(10, shuffle=True, random_state=42+r)
			pipeline = GridSearchCV(my_pipeline, parameter_grid, cv=kfolds.split(X_train, y_train), scoring=auc, n_jobs=4)
			pipeline.fit(X_train, y_train)

			pred_test = pipeline.predict(X_test)

			test_auc = auc(pipeline, X_test, y_test)

			cv_scores.append(pipeline.best_score_)
			test_scores.append(test_auc)

		std_loss = np.std(cv_scores)
		loss = np.average(cv_scores)
	else:
		kfolds = StratifiedKFold(10, shuffle=True, random_state=42)
		pipeline = GridSearchCV(my_pipeline, parameter_grid, cv=kfolds.split(X_train, y_train), scoring=auc, n_jobs=4)
		pipeline.fit(X_train, y_train)

		pred_test = pipeline.predict(X_test)

		test_auc = auc(pipeline, X_test, y_test)

		std_loss = pipeline.cv_results_['std_test_score'][pipeline.best_index_]
		#std_loss = np.min([pipeline.cv_results_['split' + str(split)+ '_test_score'][pipeline.best_index_] for split in range(10)])
		loss = pipeline.cv_results_['mean_test_score'][pipeline.best_index_]
		test_scores.append(test_auc)

	return loss, np.average(test_scores), pred_test, std_loss


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--features', nargs='+', type=int, required=True)

	args = parser.parse_args()

	score = run_pipeline(args.features)

	print(score)

if __name__ == "__main__":
	main()



