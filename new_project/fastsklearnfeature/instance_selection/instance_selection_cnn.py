from imblearn.under_sampling import CondensedNearestNeighbour


def sample_data_by_cnn(X, y):
    cnn = CondensedNearestNeighbour(random_state=42)
    return cnn.fit_resample(X, y)