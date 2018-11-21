from ml.kaggle.representation_learning.Transformer.Transformer import Transformer
import numpy as np
from ml.kaggle.Regression.library import get_all_transformations_per_column
from ml.kaggle.classifier.xgboost.LogisticRegressionClassifier import LogisticRegressionClassifier
from ml.kaggle.score.F1Score import F1Score
import copy


class RandomSearch:
    def __init__(self, pandas_dataframe, target_column, number_of_valid_configs, Score=F1Score, Classifier=LogisticRegressionClassifier):
        self.pandas_dataframe = pandas_dataframe
        self.target_column = target_column
        self.Score = Score
        self.Classifier = Classifier
        self.number_of_valid_configs = number_of_valid_configs


    def run(self):

        print self.pandas_dataframe.shape

        transformer = Transformer(self.pandas_dataframe, self.target_column, map=False, number_clusters_for_target=1)
        transformer.create_train_test_valid_stratified(train_fraction=[0.66, 1000000], valid_fraction=0.0, test_fraction=0.44, seed=42)

        transformations = get_all_transformations_per_column(self.pandas_dataframe, self.target_column)

        rand_state = np.random.RandomState(seed=42)

        my_Score = self.Score(transformer.number_classes)
        classifier = self.Classifier(transformer.number_classes, my_Score)

        apply_hyperparameter_optimization = False
        cross_val_folds = 10

        self.fscore = []
        self.fscore_best = []

        N_runs = 0
        while True:

                transformers = []

                #we randomly choose one feature representation per attribute or none
                for col_i in range(self.pandas_dataframe.shape[1]):
                    if col_i != self.target_column:
                        best_i = rand_state.randint(len(transformations[col_i]) + 1)
                        if best_i != len(transformations[col_i]):
                            transformer_default = copy.deepcopy(transformations[col_i][best_i])
                            transformer_default.column_id = col_i
                            transformers.append(transformer_default)
                        else: #skip attribute
                            print "skip attribute"


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
        
                '''
                for t_i in transformers:
                    try:
                        print t_i.__class__.__name__ + ": " + str(len(t_i.get_feature_names(self.pandas_dataframe)))
                    except:
                        print t_i.__class__.__name__ + ": " + "exception"



                assert datasets[0].shape[1] == len(feature_names), "Feature names does not fit to data dimensions"


                try:
                    y_pred = default_model.predict(datasets[1]) #test
                    current_score = my_Score.score(targets[1], y_pred) #test


                    if apply_hyperparameter_optimization:
                        y_pred_best = model_hyperparameter_optimized.predict(datasets[1]) #test
                        current_score_best = my_Score.score(targets[1], y_pred_best) #test

                    self.fscore.append(current_score)

                    if apply_hyperparameter_optimization:
                        self.fscore_best.append(current_score_best)

                    print transformer.print_config()

                    print "default F1: " + str(current_score)
                    print "max: " + str(np.max(self.fscore))
                    if apply_hyperparameter_optimization:
                        print "optimzed F1: " + str(current_score_best)
                    N_runs += 1

                except Exception as e:
                    print e

                if N_runs == self.number_of_valid_configs:
                    break

if __name__ == '__main__':
    from ml.kaggle.datasets.data_collection import get_data
    import pandas as pd
    dataset = get_data()[0]
    dataframe = pd.read_csv(dataset[0])
    search = RandomSearch(dataframe, dataset[1], 1)
    search.run()