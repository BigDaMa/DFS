from ml.kaggle.Regression.read.TranformerResult import TransformerResult
from ml.kaggle.representation_learning.Transformer.Transformer import Transformer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

results = []

t = None

#with open('/home/felix/FastFeatures/log/log_features_svm.csv') as f:
with open('/home/felix/FastFeatures/log/log_features_xgboost.csv') as f:
#with open('/home/felix/FastFeatures/log/log_features_naive_bayes.csv') as f:
#with open('/home/felix/FastFeatures/log/log_features_logistic_regression.csv') as f:
#with open('/home/felix/FastFeatures/log/log_features_knn.csv') as f:
    for line in f:
        data = line.split(":")
        column_id = int(data[0])
        column_name = data[1]
        transformer_name = data[2][1:len(data[2])]
        fscore = float(data[3])

        if type(t) == type(None):
            t = TransformerResult(column_id)
        else:
            if t.column_id != column_id:
                results.append(t)
                t = TransformerResult(column_id)

        t.add_result(transformer_name, fscore)

        #print line

#print results


def calculate_attribute_importance(importance_scores, featurenames):
    attribute_importance = {}

    for i_feature in range(len(importance_scores)):
        attribute_id = int(feature_names[i_feature].split("#")[0])
        if not attribute_id in attribute_importance:
            attribute_importance[attribute_id] = importance_scores[i_feature]
        else:
            attribute_importance[attribute_id] += importance_scores[i_feature]

    return attribute_importance




pandas_table = pd.read_csv("/home/felix/.kaggle/datasets/dansbecker/melbourne-housing-snapshot/melb_data.csv", encoding="utf-8", parse_dates=True)
transformer = Transformer(pandas_table, 4)

fscore = []

while True:

        transformers = []
        skip_columns = []

        for result in results:
            if not result.column_id in skip_columns:

                #best_i = np.random.randint(len(result.transformers) + 1)
                best_i = 0
                if best_i != len(result.transformers):
                    transformers.append(result.get_best_transformer(best_i))


        transformer.transformers = transformers
        transformer.fit()
        datasets, targets, feature_names = transformer.transform()

        regr = xgb.XGBClassifier(objective='multi:softprob', nthread=4)
        regr.fit(datasets[0], targets[0])

        # get feature importance
        b = regr.get_booster()
        fs = b.get_score('', importance_type='gain')
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        sorted = np.argsort(-all_features)

        r_result = calculate_attribute_importance(all_features[sorted], np.array(feature_names)[sorted])

        import operator
        key = max(r_result.iteritems(), key=operator.itemgetter(1))[0]
        print key

        break

        print str(r_result)

        assert len(all_features) == len(feature_names), "Oh no! This assertion failed!"

        number_of_features = 10
        show_features = np.array(feature_names)[sorted][0:number_of_features]
        #print np.array(feature_names)[sorted]

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




        #from sklearn import svm
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

        y_pred = regr.predict(datasets[1])

        current_score = f1_score(targets[1], y_pred, average='micro')
        fscore.append(current_score)

        print transformer.print_config()

        print "F1: " + str(current_score)
        print "max: " + str(np.max(fscore))

print str(list(fscore))