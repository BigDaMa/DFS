import numpy as np
from sklearn.metrics import make_scorer

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder



def micro_auc(y_true, y_score, sample_weight=None):
    print(y_true)
    print(y_score)

    y_true = OneHotEncoder(handle_unknown='ignore').fit_transform(y_true.reshape(-1, 1))
    y_score = OneHotEncoder(handle_unknown='ignore').fit_transform(y_score.reshape(-1, 1))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

#y = classifier.fit(X_train, y_train).predict(X_test)
#print(y)

#print(micro_auc(y_test, y))

print(cross_validate(classifier, iris.data, iris.target, cv=5, scoring=make_scorer(micro_auc)))