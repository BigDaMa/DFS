import numpy as np
from scipy import stats


def precision_at_k(actual,prediction, k):
    top_k = np.where(stats.rankdata(np.array(actual) * -1, method='dense') <= k)[0]
    predicted_top_k = np.where(stats.rankdata(np.array(prediction) * -1, method='dense') <= k)[0]

    return len(set(top_k) & set(predicted_top_k)) / float(len(predicted_top_k))

#print precision_at_k([0.1, 0.3, 0.2, 0.4], [0.6, 0.5, 0.1, 0.2], 3)


print precision_at_k([0.6, 0.3, 0.6, 0.5, 0.7], [0.1, 0.2, 0.3, 0.1, 0.5], 3)