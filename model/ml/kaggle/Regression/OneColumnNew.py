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


import  kaggle.representation_learning.Transformer.TransformerImplementations.all as alltransform
import kaggle.representation_learning.Transformer.TransformerImplementations.categorical as cattransform
import kaggle.representation_learning.Transformer.TransformerImplementations.numeric as numtransform
from kaggle.representation_learning.Transformer.TransformerImplementations.parser.DateTransformer import DateTransformer


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





numerical_transformers = [numtransform.BinarizerTransformer(0),
                        numtransform.BucketTransformer(0),
                        numtransform.ClusterDistTransformer(0),
                        numtransform.ClusterTransformer(0),
                        numtransform.CosTransformer(0),
                        numtransform.DegreesTransformer(0),
                        numtransform.DummyTransformer(0),
                        numtransform.IdentityTransformer(0),
                        numtransform.ImputerTransformer(0),
                        numtransform.LogTransformer(0),
                        numtransform.PlottingPositionTransformer(0),
                        numtransform.PolynomialTransformer(0),
                        numtransform.QuantileTransformer(0),
                        numtransform.RadiansTransformer(0),
                        numtransform.RankTransformer(0),
                        numtransform.RSHTransformer(0),
                        numtransform.ScaleTransformer(0),
                        numtransform.SigmoidTransformer(0),
                        numtransform.SinTransformer(0),
                        numtransform.SqrtTransformer(0),
                        numtransform.SquareTransformer(0),
                        numtransform.TanTransformer(0),
                        numtransform.ToIntTransformer(0),
                        numtransform.TrimtailTransformer(0),
                        numtransform.WinsorizeTransformer(0),
                        numtransform.ZScoreTransformer(0)]
categorical_transformers = [cattransform.OneHotTransformer(0),
                        cattransform.FrequencyEncodingTransformer(0),
                        cattransform.OrdinalTransformer(0)]
all_transformers = [#alltransform.AvgWord2VecTransformer(0),
                        alltransform.HashingTransformer(0),
                        alltransform.LengthCountTransformer(0),
                        alltransform.NgramTransformer(0, analyzer='char'),
                        alltransform.NgramTransformer(0, analyzer='word'),


                        alltransform.ParseNumbersTransformer(0),
                        DateTransformer(0)
                   ]



log_file = open('/tmp/log_features.csv', 'w+')

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

        print mypath
        print target_column

        pandas_table = pd.read_csv(mypath, encoding="utf-8", parse_dates=True)
        #pandas_table = pandas_table.fillna('0.0')

        import time

        start = time.time()

        for col_rep in range(pandas_table.shape[1]):

            if col_rep != target_column:

                transformers = []
                transformers.extend(all_transformers)

                # check if numerical
                if pandas_table.dtypes[col_rep] == 'int64' or pandas_table.dtypes[col_rep] == 'float64':
                    transformers.extend(numerical_transformers)

                # check if categorical
                count = len(pandas_table[pandas_table.columns[col_rep]].unique())
                fraction = count / float(len(pandas_table))
                if fraction < 0.05:
                    transformers.extend(categorical_transformers)

                new_dataframe = pandas_table[[pandas_table.columns[col_rep], pandas_table.columns[target_column]]]

                print new_dataframe.shape

                #new_dataframe = new_dataframe.sample(n=1500)
                #new_dataframe.reset_index(inplace=True, drop=True)

                for current_transformer in transformers:

                    transformer = Transformer(new_dataframe, 1, map=False)
                    transformer.create_train_test(25, 2000)


                    transformer.transformers = [current_transformer]
                    transformer.fit()

                    datasets, targets, feature_names = transformer.transform()


                    if type(datasets[0]) == type(None):
                        log_file.write(str(col_rep) + ": " + str(pandas_table.columns[col_rep]) + ": " + str(
                            transformer.transformers[0]) + ": " + str(0.0) + "\n")
                        log_file.flush()

                        continue

                    assert datasets[0].shape[1] == datasets[1].shape[1] and datasets[1].shape[1] == datasets[2].shape[1]

                    print str(type(datasets[0]))



                    print str(datasets[0].shape)

                    if len(datasets[0].shape) == 1:
                        for data_i in range(3):
                            datasets[data_i] = np.matrix(datasets[data_i]).T

                    regr = xgb.XGBClassifier(objective='multi:softprob', nthread=4)
                    regr.fit(datasets[0], targets[0])

                    from sklearn import svm
                    #regr = svm.SVC()
                    #regr.fit(datasets[0], targets[0])

                    from sklearn.naive_bayes import MultinomialNB
                    #regr = MultinomialNB()
                    #regr.fit(np.abs(datasets[0]), targets[0])

                    from sklearn.linear_model import LogisticRegression
                    #regr = LogisticRegression()
                    #regr.fit(datasets[0], targets[0])

                    from sklearn.neighbors import KNeighborsClassifier
                    #regr = KNeighborsClassifier()
                    #regr.fit(datasets[0], targets[0])


                    '''  
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
    
                    number_of_features = 10
                    show_features = np.array(feature_names)[sorted][0:number_of_features]
                    
                    # Visualize model                    
                    fig, ax = plt.subplots()
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
                    '''


                    print "##############"

                    #transformer.print_config()
                    #print transformer.attribute_position

                    # Make predictions using the testing set
                    y_pred = regr.predict(datasets[1])

                    print "F1: " + str(f1_score(targets[1], y_pred, average='micro'))

                    log_file.write(str(col_rep) + ": " + str(pandas_table.columns[col_rep]) + ": " + str(transformer.transformers[0]) + ": " + str(f1_score(targets[1], y_pred, average='micro')) + "\n")
                    log_file.flush()

        end = time.time()
        print("time: " + str(end - start))

        #print explain_prediction_me(X_test[0, :], regr, feature_names)

        '''
        y_test = targets[1]
        for record_i in [0]: #range(len(y_test)):
            if y_test[record_i] != y_pred[record_i]:
                print "record id: " + str(record_i)
                print "Actual class: " + str(y_test[record_i])
                print "Predicted class: " + str(y_pred[record_i])


                one_row = np.matrix(datasets[1][record_i, :]).A1

                #explain_prediction_me(one_row, regr, feature_names)
                break
        '''


        break

log_file.close()