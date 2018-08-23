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




all_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_all.csv', encoding="utf-8", sep=':')
random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_500.csv', encoding="utf-8", sep=':')
#random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_1000.csv', encoding="utf-8", sep=':')
#random_sample_records = pd.read_csv('/home/felix/phd/ranking_test/log_features_1500.csv', encoding="utf-8", sep=':')



all = get_results_per_column(all_records)
sample = get_results_per_column(random_sample_records)

taus = []
rhos = []
for key, value in all.iteritems():
    '''
    if len(value)>=8:
        z, pval = stats.normaltest(value)
        print stats.normaltest(value)
        if (pval < 0.055):
            print "Not normal distribution"
    '''


    t = stats.kendalltau(value, sample[key])[0] # higher complexity n^2, better statistical properties
    r = stats.spearmanr(value, sample[key])[0]
    taus.append(t)
    rhos.append(r)
    print "col: " + str(key) + " tau: " + str(t) + " rho: " + str(r)

print np.mean(np.array(taus))
#print np.mean(np.array(rhos))