from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from  sklearn.cluster import MiniBatchKMeans
from library.cluster_selection.UCB1_ZOMBIE import UCB1
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sets import Set
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


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

n_clusters = 50#500

#kmeans = KMeans(n_clusters=500, random_state=0).fit(x)
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


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'probability': [True]},
                    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'probability': [True]}]



newsgroups_test = fetch_20newsgroups(subset='test')

X_test = pipeline.transform(newsgroups_test.data)
y_test = newsgroups_test.target

X_train = None
y_train = []

clf = None


fscore_list = [0]


for t in range(150):
    #select cluster to draw from
    while(True):
        selected_cluster = algo.select_arm()
        try:
            record_id = clusters[selected_cluster].pop()
            break
        except KeyError:
            algo.update(selected_cluster, -100000)

    if X_train == None:
        X_train = x[record_id]
    else:
        X_train = vstack((X_train, x[record_id]))
    y_train.append(y[record_id])

    if t > 50:
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=4)

        clf.fit(X_train, y_train)

        print clf.best_params_
        clf_all = SVC(**clf.best_params_)
        clf_all.fit(X_train, y_train)


        print "time: " + str(t)
        print "Number classes: " + str(len(clf.classes_))



        y_true, y_pred = y_test, clf.predict(X_test)

        fscore_list.append(accuracy_score(y_true, y_pred))

        algo.update(selected_cluster, fscore_list[-1] - fscore_list[-2])

        print(fscore_list[-1])
    else:
        algo.update(selected_cluster, 0.001)

fig, ax = plt.subplots()
ax.plot(range(len(fscore_list)), fscore_list, label='Accuracy')
plt.show()