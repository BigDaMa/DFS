from skfeature.function.similarity_based.fisher_score import fisher_score
import copy


def my_fisher_score(X, y):
    return fisher_score(copy.deepcopy(X), y.flatten())