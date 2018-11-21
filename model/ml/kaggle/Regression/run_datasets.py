from ml.kaggle.datasets.data_collection import get_data
from ml.kaggle.Regression.random_search import RandomSearch
from ml.kaggle.Regression.all_one_column_search import SingleColumnSearch
import pandas as pd
from ml.kaggle.Regression.plot.histogram_latex import plot
import numpy as np
from ml.kaggle.classifier.xgboost.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml.kaggle.classifier.xgboost.KNearestNeighborClassifier import KNearestNeighborClassifier
from ml.kaggle.score.RocAUCScore import RocAUCScore



datasets = get_data()

#run random search

fscore_results = []
single_column_results = []

#classifier = KNearestNeighborClassifier
classifier = LogisticRegressionClassifier
score = RocAUCScore

for d_i in range(len(datasets)):
    dataset = get_data()[d_i]
    dataframe = pd.read_csv(dataset[0])
    search = RandomSearch(dataframe, dataset[1], 250, Classifier=classifier, Score=score)
    search.run()
    fscore_results.append(search.fscore)

    single_column_search = SingleColumnSearch(dataframe, dataset[1], 10, Classifier=classifier, Score=score)
    single_column_search.run()
    single_column_results.append(np.mean(single_column_search.fscore_list))

    data = plot([search.fscore], [np.mean(single_column_search.fscore_list)], [datasets[d_i]])

    my_file = open('/tmp/histogram_logistic_regression' + str(d_i) +'.tex', 'w+')
    my_file.write(data)
    my_file.close()



