from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sets import Set
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score
from library.initialisation.InitialTraining import InitialTraining
import time

import numpy as np
import matplotlib.pyplot as plt

from library.stat_model.SVC_Model import SVC_Model
from library.Data.News import News
from library.cluster_selection.ClusterFrequency import ClusterFrequency


def train_model(X_train, y_train, stat_model):
    #stat_model.optimize_hyperparameters(X_train, y_train, folds=5)
    # params = stat_model.best_params

    params = {'kernel': 'rbf', 'C': 10, 'probability': True, 'gamma': 0.1}
    #params = {'kernel': 'rbf', 'C': 1, 'probability': True, 'gamma': 0.001} #with scale
    print params

    stat_model.train(X_train, y_train, params)




def test_model(stat_model, X_test, y_test, fscore_list):
    y_true, y_pred = y_test, stat_model.predict(X_test)

    fscore_list.append(accuracy_score(y_true, y_pred))
    print(fscore_list[-1])

# data
data = News()
#data = Spam()
data_train_x, data_train_y = data.get_train()
data_test_x, data_test_y = data.get_test()


random_seed = 42


# specify TF-IDF
analyzer="word"
ngrams = 1
pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer,
                                              lowercase=False,
                                              ngram_range=(1, ngrams))),
                     ('tfidf', TfidfTransformer()),
                     #('scale', StandardScaler(with_mean=False))
                    ])


pipeline.fit(data_train_x)

x = pipeline.transform(data_train_x)
y = data_train_y

X_test = pipeline.transform(data_test_x)
y_test = data_test_y

print x.shape[0]


'''
y_encoder = LabelBinarizer()
y_encoder.fit(y)
y_one_hot = y_encoder.transform(y)

print y_one_hot.shape
print x.shape

all_data = hstack((x, y_one_hot))
print all_data.shape
'''

all_list = []
time_list_all = []

for run in range(10):
    random_seed += 1

    # clustering
    cluster_time = time.time()

    n_clusters = 70#500

    #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
    all_data = x
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_seed, max_iter=20, init_size=3*n_clusters).fit(all_data)
    group_labels = kmeans.predict(all_data)


    clusters = {}

    for i in range(n_clusters):
        clusters[i] = Set()

    for i in range(x.shape[0]):
        clusters[group_labels[i]].add(i)

    print("cluster time: --- %s seconds ---" % (time.time() - cluster_time))

    # bandits
    algo = ClusterFrequency(clusters)

    #incremental training

    stat_model = SVC_Model()

    X_train = None
    y_train = []

    accuracy_list = [0]
    time_list = []



    start_time = time.time()



    # create in initial training set
    init = InitialTraining(x, y, n=10, random_seed=random_seed)
    X_train, y_train = init.generate()

    # first model

    train_model(X_train, y_train, stat_model)
    test_model(stat_model, X_test, y_test, accuracy_list)
    time_list.append((time.time() - start_time))

    #define reward
    #reward = RecordUncertainty(stat_model, data.get_number_classes())
    #reward = RecordError(stat_model)

    for t in range(n_clusters):
        #select cluster to draw from
        while(True):
            selected_cluster = algo.select_arm()
            print "select cluster: " + str(selected_cluster)
            try:
                record_id = clusters[selected_cluster].pop()
                break
            except KeyError:
                print "warn"

        X_train = vstack((X_train, x[record_id]))
        y_train.append(y[record_id])


        print "time: " + str(t)

        train_model(X_train, y_train, stat_model)
        test_model(stat_model, X_test, y_test, accuracy_list)

        time_list.append((time.time() - start_time))

    print("--- %s seconds ---" % (time.time() - start_time))

    all_list.append(accuracy_list)
    time_list_all.append(time_list)

mean_list = list(np.mean(all_list, axis=0))
mean_time = list(np.mean(time_list_all, axis=0))

fig, ax = plt.subplots()
ax.plot(range(len(mean_list)), mean_list, label='Accuracy')
plt.show()

print list(mean_list)
print list(mean_time)