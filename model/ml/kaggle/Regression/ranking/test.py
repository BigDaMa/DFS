import pandas as pd
from sklearn.metrics import label_ranking_loss
import numpy as np
from scipy import stats

def get_results_per_column(df):
    results_per_column = {}
    for i in range(df.shape[0]):
        if not df.values[i, 0] in results_per_column:
            results_per_column[df.values[i, 0]] = []
        results_per_column[df.values[i, 0]].append(float(df.values[i, 3]))

    return results_per_column


def precision_at_k(actual,prediction, k):
    top_k = np.where(stats.rankdata(np.array(actual) * -1, method='dense') <= k)[0]
    predicted_top_k = np.where(stats.rankdata(np.array(prediction) * -1, method='dense') <= k)[0]

    return len(set(top_k) & set(predicted_top_k)) / float(len(predicted_top_k))




all_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_10000.csv', encoding="utf-8", sep=':')
#random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_500.csv', encoding="utf-8", sep=':')
#random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_1000.csv', encoding="utf-8", sep=':')
#random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_1500.csv', encoding="utf-8", sep=':')

#all_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_test09.1.csv', encoding="utf-8", sep=':')
random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_1000.csv', encoding="utf-8", sep=':')



all = get_results_per_column(all_records)
sample = get_results_per_column(random_sample_records)

taus = []
rhos = []
precisionatks = []
for key, value in all.iteritems():
    '''
    if len(value)>=8:
        z, pval = stats.normaltest(value)
        print stats.normaltest(value)
        if (pval < 0.055):
            print "Not normal distribution"
    '''


    precAt3 = precision_at_k(value, sample[key], 1)

    t = stats.kendalltau(value, sample[key])[0] # higher complexity n^2, better statistical properties
    r = stats.spearmanr(value, sample[key])[0]
    taus.append(t)
    rhos.append(r)
    precisionatks.append(precAt3)
    print "col: " + str(key) + " tau: " + str(t) + " rho: " + str(r) + " prec@3: " + str(precAt3)


print np.mean(np.array(taus))
#print np.mean(np.array(rhos))
print np.mean(np.array(precisionatks))