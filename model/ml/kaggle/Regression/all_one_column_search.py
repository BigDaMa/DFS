from ml.kaggle.Regression.read.TranformerResult import TransformerResult
from ml.kaggle.representation_learning.Transformer.Transformer import Transformer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
import xgboost as xgb
import numpy as np
from ml.kaggle.Regression.library import get_all_transformations
from ml.kaggle.classifier.xgboost.XGBoostClassifier import XGBoostClassifier
from ml.kaggle.classifier.xgboost.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml.kaggle.classifier.xgboost.KNearestNeighborClassifier import KNearestNeighborClassifier
from ml.kaggle.score.F1Score import F1Score
from ml.kaggle.score.RocAUCScore import RocAUCScore
import pickle
import copy



results = []

t = None

'''
#read results from previous runs to know which transformers are applicable
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
'''

#kaggle data


#read data
#pandas_table = pd.read_csv("/home/felix/.kaggle/datasets/dansbecker/melbourne-housing-snapshot/melb_data.csv", encoding="utf-8")
#target_colum = 4

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv")
target_colum = 13

'''
pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv")
target_colum = 23

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv")
target_colum = 30

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv")
target_colum = 10

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv")
target_colum = 15

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv")
target_colum = 8

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv")
target_colum = 20

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv")# 3 classes
target_colum = 9

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/phpkIxskf_bank_data.csv")
target_colum = 16

pandas_table = pd.read_csv("/home/felix/datasets/ExploreKit/csv/vehicleNorm.csv")
target_colum = 100
'''

transformer = Transformer(pandas_table, target_colum, map=False, number_clusters_for_target=1)

fscore = []
fscore_best = []


transformations = get_all_transformations()

rand_state = np.random.RandomState(seed=42)


my_Score = F1Score(transformer.number_classes)
#my_Score = RocAUCScore(transformer.number_classes)


classifier = LogisticRegressionClassifier(transformer.number_classes, my_Score)
#classifier = XGBoostClassifier(transformer.number_classes, my_Score)
#classifier = KNearestNeighborClassifier(transformer.number_classes, my_Score)

apply_hyperparameter_optimization = False
cross_val_folds = 10

final_transformers = []
transformed_columns = []



for attribute_i in range(pandas_table.shape[1]):
    if attribute_i != target_colum:
        attribute_scores = []
        attribute_transformation = []
        for transformation_i in range(len(transformations)):

            transformers = []
            transformer_default = copy.deepcopy(transformations[transformation_i])
            transformer_default.column_id = attribute_i
            transformers.append(transformer_default)


            transformer.transformers = transformers
            failed_transformation = False
            try:
                transformer.fit()
                datasets, targets, feature_names = transformer.transform()

                if apply_hyperparameter_optimization:
                    best_params = classifier.run_cross_validation(datasets[0], targets[0], cross_val_folds)
                    model_hyperparameter_optimized = classifier.fit(datasets[0], targets[0], best_params)

                default_model = classifier.fit(datasets[0], targets[0])
            except Exception as e:
                print e
                failed_transformation = True
            if failed_transformation:
                continue




            #apply hyperparameter tuning

            '''
            # get feature importance
            b = model_hyperparameter_optimized.get_booster()
            fs = b.get_score('', importance_type='gain')
            all_features = [fs.get(f, 0.) for f in b.feature_names]
            all_features = np.array(all_features, dtype=np.float32)
            sorted = np.argsort(-all_features)
            
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
    
    
            #for t_i in transformers:
            #    print t_i.__class__.__name__ + ": " + str(len(t_i.get_feature_names(pandas_table)))
            
            '''

            assert datasets[0].shape[1] == len(feature_names), "Feature names does not fit to data dimensions"



            try:
                y_pred = default_model.predict(datasets[1])
                current_score = my_Score.score(targets[1], y_pred)


                if apply_hyperparameter_optimization:
                    y_pred_best = model_hyperparameter_optimized.predict(datasets[1])
                    current_score_best = my_Score.score(targets[1], y_pred_best)

                fscore.append(current_score)

                if apply_hyperparameter_optimization:
                    fscore_best.append(current_score_best)

                attribute_scores.append(current_score)
                print attribute_scores

                attribute_transformation.append(transformation_i)


                print transformer.print_config()

                print "default F1: " + str(current_score)
                print "max: " + str(np.max(fscore))
                if apply_hyperparameter_optimization:
                    print "optimzed F1: " + str(current_score_best)


            except Exception as e:
                print e

        #add best to final transformation
        max_id = np.argmax(np.array(attribute_scores))
        final_transformers.append(attribute_transformation[max_id])
        transformed_columns.append(attribute_i)

print "fscore"
print str(list(fscore))
print "best fscore"
print str(list(fscore_best))

file_pi = open('/tmp/final_transformations.obj', 'w+')
pickle.dump(final_transformers, file_pi)
file_pi.close()

file_pi = open('/tmp/final_column_ids.obj', 'w+')
pickle.dump(transformed_columns, file_pi)
file_pi.close()
