from ml.kaggle.representation_learning.Transformer.Transformer import Transformer
import numpy as np
from ml.kaggle.Regression.library import get_all_transformations
from ml.kaggle.classifier.xgboost.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml.kaggle.score.F1Score import F1Score
import copy



class SingleColumnSearch:
    def __init__(self, pandas_dataframe, target_column, number_runs, Score=F1Score, Classifier=LogisticRegressionClassifier):
        self.pandas_dataframe = pandas_dataframe
        self.target_column = target_column
        self.Score = Score
        self.Classifier = Classifier
        self.number_runs = number_runs


    def run(self):

        transformer = Transformer(self.pandas_dataframe, self.target_column, map=False, number_clusters_for_target=1)

        self.fscore_list = []


        transformations = get_all_transformations()
        rand_state = np.random.RandomState(seed=42)


        my_Score = self.Score(transformer.number_classes)
        classifier = self.Classifier(transformer.number_classes, my_Score)

        apply_hyperparameter_optimization = False
        cross_val_folds = 10


        for run_i in range(self.number_runs):
            transformer.create_train_test_valid(train_fraction=[0.6, 1000], valid_fraction=0.2, test_fraction=0.2, seed=42 + run_i)

            final_transformers = []
            transformed_columns = []

            for attribute_i in range(self.pandas_dataframe.shape[1]):
                if attribute_i != self.target_column:
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

                            attribute_scores.append(current_score)
                            attribute_transformation.append(transformation_i)

                        except Exception as e:
                            print e

                    #add best to final transformation
                    max_id = np.argmax(np.array(attribute_scores))
                    final_transformers.append(attribute_transformation[max_id])
                    transformed_columns.append(attribute_i)

            #apply best single configurations together
            transformer_indices = final_transformers
            transformed_column_indices = transformed_columns

            transformers = []
            all_transformations = get_all_transformations()
            for t_i in range(len(transformer_indices)):
                transformer_default = copy.deepcopy(all_transformations[transformer_indices[t_i]])
                transformer_default.column_id = transformed_column_indices[t_i]
                transformers.append(transformer_default)

            transformer.transformers = transformers

            transformer.fit()
            datasets, targets, feature_names = transformer.transform()



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


                self.fscore_list.append(current_score)
            except:
                self.fscore_list.append(0.0)

if __name__ == '__main__':
    from ml.kaggle.datasets.data_collection import get_data
    import pandas as pd
    dataset = get_data()[0]
    dataframe = pd.read_csv(dataset[0])
    search = SingleColumnSearch(dataframe, dataset[1], 1)
    search.run()
    print search.fscore_list
