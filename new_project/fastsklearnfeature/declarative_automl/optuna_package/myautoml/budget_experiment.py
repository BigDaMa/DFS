from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLProcess import MyAutoML
import optuna
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml


from fastsklearnfeature.declarative_automl.optuna_package.myautoml.Space_GenerationTree import SpaceGenerator

auc=make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

#dataset = openml.datasets.get_dataset(40536)
dataset = openml.datasets.get_dataset(31)
#dataset = openml.datasets.get_dataset(1590)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1, stratify=y, train_size=0.6)

gen = SpaceGenerator()
space = gen.generate_params()

print('number hyperparameters: ' + str(len(space.name2node)))

from anytree import Node, RenderTree

for pre, _, node in RenderTree(space.parameter_tree):
    print("%s%s: %s" % (pre, node.name, node.status))

my_study = optuna.create_study(direction='maximize')

validation_scores = []
test_scores = []

#add Caruana ensemble with replacement # save pipelines to disk

for i in range(1, 10):
    search = MyAutoML(cv=10, number_of_cvs=1, n_jobs=2, time_search_budget=2*60, space=space, study=my_study, main_memory_budget_gb=4)
    best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)
    my_study = search.study

    test_score = auc(search.get_best_pipeline(), X_test, y_test)

    print("budget: " + str(i) + ' => ' + str(best_result) + " test: " + str(test_score))

    validation_scores.append(best_result)
    test_scores.append(test_score)


print(validation_scores)
print(test_scores)

