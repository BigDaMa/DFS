from skfeature.function.similarity_based.fisher_score import fisher_score


def my_fisher_score(X, y):
    return fisher_score(X, y.flatten())