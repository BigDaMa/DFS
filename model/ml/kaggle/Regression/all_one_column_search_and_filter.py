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


number_runs = 10



fscore = []
fscore_best = []


transformations = get_all_transformations()

rand_state = np.random.RandomState(seed=42)


my_Score = F1Score(transformer.number_classes)


classifier = LogisticRegressionClassifier(transformer.number_classes, my_Score)
#classifier = XGBoostClassifier(transformer.number_classes, my_Score)
#classifier = KNearestNeighborClassifier(transformer.number_classes, my_Score)

apply_hyperparameter_optimization = False
cross_val_folds = 10

fscore_list = []


for run_i in range(number_runs):
    transformer.create_train_test_valid(train_fraction=0.6, valid_fraction=0.2, test_fraction=0.2, seed=42 + run_i)

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

                assert datasets[0].shape[1] == len(feature_names), "Feature names does not fit to data dimensions"



                try:
                    y_pred = default_model.predict(datasets[2]) #check validation
                    current_score = my_Score.score(targets[2], y_pred)


                    if apply_hyperparameter_optimization:
                        y_pred_best = model_hyperparameter_optimized.predict(datasets[2])#check validation
                        current_score_best = my_Score.score(targets[2], y_pred_best)

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



    filehandler = open('/tmp/final_transformations.obj', 'r')
    transformer_indices = pickle.load(filehandler)
    filehandler.close()
    filehandler = open('/tmp/final_column_ids.obj', 'r')
    transformed_column_indices = pickle.load(filehandler)
    filehandler.close()
    transformers = []
    all_transformations = get_all_transformations()
    for t_i in range(len(transformer_indices)):
        transformer_default = copy.deepcopy(all_transformations[transformer_indices[t_i]])
        transformer_default.column_id = transformed_column_indices[t_i]
        transformers.append(transformer_default)

    transformer.transformers = transformers

    transformer.fit()
    datasets, targets, feature_names = transformer.transform()

    #apply feature selection
    from sklearn.feature_selection import RFE

    feature_number_2_score = {}
    for feature_i in range(1, datasets[0].shape[1]):
        rfecv = RFE(estimator=classifier.get_classifier(), step=1, n_features_to_select=feature_i)
        rfecv.fit(datasets[0], targets[0])

        new_data = rfecv.transform(datasets[0])
        new_data_validation = rfecv.transform(datasets[2])

        default_model = classifier.fit(new_data, targets[0])

        y_pred = default_model.predict(new_data_validation)  # validation
        feature_number_2_score[feature_i] = my_Score.score(targets[2], y_pred)  # validation

    max_value = -1
    best_features = -1
    for feature_i in range(1, datasets[0].shape[1]):
        if feature_number_2_score[feature_i] > max_value:
            best_features = feature_i
            max_value = feature_number_2_score[feature_i]

    perfect_number_feature = best_features

    print "original: " + str(datasets[0].shape[0])
    print "selected: " + str(perfect_number_feature)

    rfecv = RFE(estimator=classifier.get_classifier(), step=1, n_features_to_select=perfect_number_feature)
    rfecv.fit(datasets[0], targets[0])

    datasets[0] = rfecv.transform(datasets[0])
    datasets[1] = rfecv.transform(datasets[1])
    datasets[2] = rfecv.transform(datasets[2])


    '''
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit(datasets[0])

    datasets[0] = sel.transform(datasets[0])
    datasets[1] = sel.transform(datasets[1])
    datasets[2] = sel.transform(datasets[2])
    '''




    #print rfe.ranking_
    #break



    if apply_hyperparameter_optimization:
        best_params = classifier.run_cross_validation(datasets[0], targets[0], cross_val_folds)
        model_hyperparameter_optimized = classifier.fit(datasets[0], targets[0], best_params)

    default_model = classifier.fit(datasets[0], targets[0])

    try:
        y_pred = default_model.predict(datasets[1])  # test
        current_score = my_Score.score(targets[1], y_pred)  # test

        if apply_hyperparameter_optimization:
            y_pred_best = model_hyperparameter_optimized.predict(datasets[1])  # test
            current_score_best = my_Score.score(targets[1], y_pred_best)  # test

        fscore.append(current_score)

        if apply_hyperparameter_optimization:
            fscore_best.append(current_score_best)

        print transformer.print_config()

        print "default F1: " + str(current_score)
        fscore_list.append(current_score)
        if apply_hyperparameter_optimization:
            print "optimzed F1: " + str(current_score_best)
    except:
        fscore_list.append(0.0)


print np.mean(fscore_list)
print len(fscore_list)
print fscore_list


