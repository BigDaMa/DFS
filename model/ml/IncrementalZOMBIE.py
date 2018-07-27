from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from library.cluster_selection.UCB1_ZOMBIE import UCB1
from sets import Set
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score
from library.initialisation.InitialTraining import InitialTraining
import time

import matplotlib.pyplot as plt

from library.stat_model.SVC_Model import SVC_Model
from library.Reward.RecordUncertainty import RecordUncertainty




def train_model(X_train, y_train, stat_model):
    # model.optimize_hyperparameters(X_train, y_train, folds=5)
    params = {'kernel': 'rbf', 'C': 10, 'probability': True, 'gamma': 0.1}  # = clf.best_params_
    print params

    stat_model.train(X_train, y_train, params)




def test_model(stat_model, X_test, y_test, fscore_list):
    y_true, y_pred = y_test, stat_model.predict(X_test)

    fscore_list.append(accuracy_score(y_true, y_pred))
    print(fscore_list[-1])

# load data
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

#print newsgroups_train

analyzer="word"
ngrams = 1
pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer,
                                              lowercase=False,
                                              ngram_range=(1, ngrams))),
                     ('tfidf', TfidfTransformer())
                    ])

pipeline.fit(newsgroups_train.data)

x = pipeline.transform(newsgroups_train.data)
y = newsgroups_train.target

print x.shape[0]

# clustering

n_clusters = 100#500

#kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, max_iter=20, init_size=3*n_clusters).fit(x)
group_labels = kmeans.predict(x)


clusters = {}

for i in range(n_clusters):
    clusters[i] = Set()

for i in range(x.shape[0]):
    clusters[group_labels[i]].add(i)


# bandits
algo = UCB1(n_clusters)

#incremental training

stat_model = SVC_Model()

newsgroups_test = fetch_20newsgroups(subset='test')

X_test = pipeline.transform(newsgroups_test.data)
y_test = newsgroups_test.target

X_train = None
y_train = []

accuracy_list = [0]



start_time = time.time()



# create in initial training set
init = InitialTraining(x, y, n=2)
X_train, y_train = init.generate()

# first model

train_model(X_train, y_train, stat_model)
test_model(stat_model, X_test, y_test, accuracy_list)

#define reward
reward = RecordUncertainty(stat_model)
#reward = RecordError(stat_model)

for t in range(100):
    #select cluster to draw from
    while(True):
        selected_cluster = algo.select_arm()
        print "select cluster: " + str(selected_cluster)
        try:
            record_id = clusters[selected_cluster].pop()
            break
        except KeyError:
            algo.update(selected_cluster, -100000)

    X_train = vstack((X_train, x[record_id]))
    y_train.append(y[record_id])


    #calculate reward
    algo.update(selected_cluster, reward.get_reward(x, y, record_id))

    print "time: " + str(t)

    train_model(X_train, y_train, stat_model)
    test_model(stat_model, X_test, y_test, accuracy_list)

    # calculate reward
    #algo.update(selected_cluster, reward.get_reward(x, y, record_id))

print("--- %s seconds ---" % (time.time() - start_time))

fig, ax = plt.subplots()
ax.plot(range(len(accuracy_list)), accuracy_list, label='Accuracy')
plt.show()