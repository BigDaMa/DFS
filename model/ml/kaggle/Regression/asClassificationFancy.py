import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

import dateutil.parser
import time
import datetime
from sklearn import preprocessing
import xgboost as xgb
from scipy.sparse import hstack
from scipy import sparse
import itertools

from math import radians, cos, sin
from sklearn.cluster import KMeans
from eli5 import show_prediction
from eli5.explain import explain_prediction

from xgboost import plot_importance
from matplotlib import pyplot


from ml.kaggle.representation_learning.Transformer.Transformer import Transformer


def explain_prediction_me(x, model, feature_name_list):
    from eli5.explain import explain_prediction
    params = {}
    params['feature_names'] = feature_name_list
    #params['top'] = 5
    expl = explain_prediction(model.get_booster(), x, **params)

    #expl.targets

    for target_explanation_i in range(len(expl.targets)):
        target_explanation = expl.targets[target_explanation_i]
        print "class " + str(target_explanation.target) + " probability: " + str(target_explanation.proba) + " score: " + str(target_explanation.score)
        print "Positive:"
        for feature_weight in target_explanation.feature_weights.pos:
            print str(feature_weight.feature) + ": weight: " + str(feature_weight.weight) + " actual value: " + str(
                feature_weight.value)
        print "Negative:"
        for feature_weight in target_explanation.feature_weights.neg:
            print str(feature_weight.feature) + ": weight: " + str(feature_weight.weight) + " actual value: " + str(feature_weight.value)



with open('/home/felix/FastFeatures/kaggle/schema_2.csv') as f: #housing
#with open('/home/felix/FastFeatures/kaggle/schema_imdb.csv') as f: #housing
    for line in f:
        tokens = line.split("#")
        user = tokens[0]
        project = tokens[1]
        csv_file = tokens[2]
        type_var = tokens[3]
        task = tokens[4]
        target_column = int(tokens[5])

        mypath = "/home/felix/.kaggle/datasets/" + str(user) + "/" + str(project) + "/" + csv_file

        pandas_table = pd.read_csv(mypath, encoding="utf-8", parse_dates=True)
        #pandas_table = pandas_table.fillna('0.0')

        transformer = Transformer(pandas_table, target_column)

        for counter_it in range(3):

            transformer.fit()

            datasets, targets, feature_names = transformer.transform()



            regr = xgb.XGBClassifier(objective='multi:softprob', nthread=4)
            regr.fit(datasets[0], targets[0])

            #from sklearn import svm
            #regr = svm.SVC()
            #regr.fit(X_train, y_train)



            #get feature importance
            b = regr.get_booster()
            fs = b.get_score('', importance_type='gain')
            all_features = [fs.get(f, 0.) for f in b.feature_names]
            all_features = np.array(all_features, dtype=np.float32)
            sorted = np.argsort(-all_features)


            print len(all_features)
            print len(feature_names)
            print list(feature_names)
            assert len(all_features) == len(feature_names), "Oh no! This assertion failed!"


            # Visualize model
            number_of_features = 10
            fig, ax = plt.subplots()
            show_features = np.array(feature_names)[sorted][0:number_of_features]
            y_pos = np.arange(len(show_features))
            performance = all_features[sorted][0:number_of_features]
            ax.barh(y_pos, performance, align='center', color='green', ecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(show_features)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Gain')
            plt.show()


            transformer.print_config()
            print transformer.attribute_position


            print show_features[0]
            print show_features[0].split("#")[0]

            most_important_attribute = int(show_features[0].split("#")[0])

            transformer.next_transformation_for_attribute(most_important_attribute)

            print "##############"

            transformer.print_config()
            print transformer.attribute_position

            # Make predictions using the testing set
            y_pred = regr.predict(datasets[1])

            print "F1: " + str(f1_score(targets[1], y_pred, average='micro'))


        #print explain_prediction_me(X_test[0, :], regr, feature_names)

        y_test = targets[1]
        for record_i in [0]: #range(len(y_test)):
            if y_test[record_i] != y_pred[record_i]:
                print "record id: " + str(record_i)
                print "Actual class: " + str(y_test[record_i])
                print "Predicted class: " + str(y_pred[record_i])


                one_row = np.matrix(datasets[1][record_i, :]).A1

                #explain_prediction_me(one_row, regr, feature_names)
                break


        break