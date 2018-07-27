from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack
from library.initialisation.InitialTraining import InitialTraining
import time

import numpy as np
import matplotlib.pyplot as plt

from library.stat_model.SVC_Model import SVC_Model
from library.Data.News import News
from library.feature.TFIDF import TFIDF




def train_model(X_train, y_train, stat_model):
    stat_model.optimize_hyperparameters(X_train, y_train, folds=5)
    params = stat_model.best_params

    #params = {'kernel': 'rbf', 'C': 10, 'probability': True, 'gamma': 0.1}
    #params = {'kernel': 'rbf', 'C': 1, 'probability': True, 'gamma': 0.001} #with scale
    print params

    stat_model.train(X_train, y_train, params)




def test_model(stat_model, X_test, y_test, fscore_list):
    y_true, y_pred = y_test, stat_model.predict(X_test)

    predictive_accuracy = np.mean(y_pred == y_true)
    fscore_list.append(predictive_accuracy)
    print(predictive_accuracy)

# load data
# data
data = News()
#data = Spam()
data_train_x, data_train_y = data.get_train()
data_test_x, data_test_y = data.get_test()


random_seed = 42

'''

analyzer	char_wb
lowercase	true
max_df	1
min_df	0.01
ngrams	2
norm	l2
smooth_idf	true
stop_words	null
sublinear_tf	false
use_idf	true
'''


#print newsgroups_train
analyzer='char_wb'
lowercase=True
max_df=1.0
min_df=0.01
ngrams=2
norm="l2"
smooth_idf=True
stop_words=None
sublinear_tf=False
use_idf=True


featurizer = TFIDF(analyzer, ngrams, lowercase, stop_words, max_df, min_df, norm, use_idf, smooth_idf, sublinear_tf)
(x, y, X_test, y_test) = featurizer.featurize(data_train_x, data_train_y, data_test_x, data_test_y)


all_list = []
time_list_all = []
sum_uncertainty_all = []

#incremental training

for run in range(1):

    stat_model = SVC_Model()

    random_seed += 1

    my_random = np.random.RandomState(random_seed)

    X_train = None
    y_train = []

    accuracy_list = [0]

    #create sample for active learning
    ids = np.arange(x.shape[0])
    my_random.shuffle(ids)

    samplesize = int(x.shape[0] * 0.1)

    sample_x = x[ids[0:samplesize]]
    sample_y = y[ids[0:samplesize]]

    time_list = []

    start_time = time.time()


    ids_set = set()

    # create in initial training set
    init = InitialTraining(x, y, n=2, random_seed=random_seed)
    X_train, y_train = init.generate()

    # first model

    train_model(X_train, y_train, stat_model)
    test_model(stat_model, X_test, y_test, accuracy_list)

    uncertainty_list = []

    # start active learning
    for t in range(70):
        #predict for all records in sample
        proba_sample = stat_model.model.predict_proba(sample_x)
        #select least certain record:
        topk = np.argsort(proba_sample, axis=1)[:,proba_sample.shape[1]-2:proba_sample.shape[1]]
        max_uncertainty = -1
        record_id = -1
        sum_uncertainty = 0.0
        for s_i in range(proba_sample.shape[0]):
            if not s_i in ids_set:
                uncertainty = 1.0 - np.sum(proba_sample[s_i, topk[s_i]])
                sum_uncertainty += uncertainty
                #print uncertainty
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    record_id = s_i

        sum_uncertainty /= float(proba_sample.shape[0])

        print "uncertainty: " + str(sum_uncertainty)

        ids_set.add(record_id)
        X_train = vstack((X_train, x[record_id]))
        y_train.append(y[record_id])

        print "time: " + str(t)

        train_model(X_train, y_train, stat_model)
        test_model(stat_model, X_test, y_test, accuracy_list)

        time_list.append((time.time() - start_time))
        uncertainty_list.append(sum_uncertainty)

    all_list.append(accuracy_list)
    time_list_all.append(time_list)
    sum_uncertainty_all.append(uncertainty_list)


mean_list = list(np.mean(all_list, axis=0))
mean_time = list(np.mean(time_list_all, axis=0))
mean_uncertainty = list(np.mean(sum_uncertainty_all, axis=0))

fig, ax = plt.subplots()
ax.plot(range(len(mean_list)), mean_list, label='Accuracy')
plt.show()

print list(mean_list)
print list(mean_time)
print list(mean_uncertainty)