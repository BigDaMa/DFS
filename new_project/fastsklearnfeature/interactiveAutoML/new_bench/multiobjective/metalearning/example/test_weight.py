from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")


from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.weighted_ranking import weighted_ranking
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import TPE
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.hyperparameter_optimization import simulated_annealing
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.evolution import evolution
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.exhaustive import exhaustive
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.forward_floating_selection import forward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.backward_floating_selection import backward_floating_selection
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.recursive_feature_elimination import recursive_feature_elimination
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.strategies.fullfeatures import fullfeatures
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package.f_anova_wo import f_anova_wo

from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import chi2_score_wo
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.bench_utils import get_fair_data1_validation



X_train, X_validation, X_test, y_train, y_validation, y_test, names, sensitive_ids, key, sensitive_attribute_id = get_fair_data1_validation(dataset_key='1590')

print(X_train.shape)

weighted_ranking(X_train, X_validation, X_test, y_train, y_validation, y_test, names, sensitive_ids, ranking_functions=[variance], clf=LogisticRegression(class_weight='balanced'), min_accuracy=0.95,
			  min_fairness=0.0, min_robustness=0.0, max_number_features=1.0, log_file='/tmp/test1')

'''
weighted_ranking(X_train, X_validation, X_test, y_train, y_validation, y_test, names, sensitive_ids, ranking_functions=[chi2_score_wo], clf=RandomForestClassifier(class_weight='balanced', random_state=42), min_accuracy=0.95,
			  min_fairness=0.0, min_robustness=0.0, max_number_features=1.0, log_file='/tmp/test1')
'''
