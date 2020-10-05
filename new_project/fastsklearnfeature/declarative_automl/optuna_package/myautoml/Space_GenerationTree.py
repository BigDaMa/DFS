import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.MyAutoMLTreeSpace import MyAutoMLSpace
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna





class SpaceGenerator:
    def __init__(self):
        self.classifier_list = myspace.classifier_list
        self.private_classifier_list = myspace.private_classifier_list
        self.preprocessor_list = myspace.preprocessor_list
        self.scaling_list = myspace.scaling_list
        self.categorical_encoding_list = myspace.categorical_encoding_list

        self.space = MyAutoMLSpace()

        #generate binary or mapping for each hyperparameter


    def generate_params(self):

        self.space.generate_cat('balanced', [True, False], True)

        category_preprocessor = self.space.generate_cat('preprocessor', self.preprocessor_list, self.preprocessor_list[0])
        for p_i in range(len(self.preprocessor_list)):
            preprocessor = self.preprocessor_list[p_i]
            preprocessor.generate_hyperparameters(self.space, category_preprocessor[p_i])

        category_classifier = self.space.generate_cat('classifier', self.classifier_list, self.classifier_list[0])
        for c_i in range(len(self.classifier_list)):
            classifier = self.classifier_list[c_i]
            classifier.generate_hyperparameters(self.space, category_classifier[c_i])

        category_private_classifier = self.space.generate_cat('private_classifier', self.private_classifier_list, self.private_classifier_list[0])
        for c_i in range(len(self.private_classifier_list)):
            private_classifier = self.private_classifier_list[c_i]
            private_classifier.generate_hyperparameters(self.space, category_private_classifier[c_i])

        category_scaler = self.space.generate_cat('scaler', self.scaling_list, self.scaling_list[0])
        for s_i in range(len(self.scaling_list)):
            scaler = self.scaling_list[s_i]
            scaler.generate_hyperparameters(self.space, category_scaler[s_i])

        imputer = SimpleImputerOptuna()
        imputer.generate_hyperparameters(self.space)

        category_categorical_encoding = self.space.generate_cat('categorical_encoding', self.categorical_encoding_list, self.categorical_encoding_list[0])
        for cat_i in range(len(self.categorical_encoding_list)):
            categorical_encoding = self.categorical_encoding_list[cat_i]
            categorical_encoding.generate_hyperparameters(self.space, category_categorical_encoding[cat_i])


        return self.space

