from ml.kaggle.Regression.library import get_unary_transformations
from ml.kaggle.Regression.library import get_binary_transformations
import copy
from ml.kaggle.representation_learning.Transformer.Transformer import Transformer

class Generator:
    def __init__(self, data, y):
        self.transformations = []
        self.number_attributes = data.shape[1]
        self.data = data
        self.y = y
        self.unary_output_dimension_size = -1

        self.transformer = Transformer(self.data, self.y, map=False, number_clusters_for_target=1)
        self.transformer.create_train_test_valid_stratified()

        self.train_data = self.data.iloc[self.transformer.ids_parts[0], :]
        self.valid_data = self.data.iloc[self.transformer.ids_parts[2], :]
        self.test_data = self.data.iloc[self.transformer.ids_parts[1], :]

        self.train_data_bin = None
        self.valid_data_bin = None
        self.test_data_bin = None

        self.train_data_bin_una = None
        self.valid_data_bin_una = None
        self.test_data_bin_una = None

        self.train_data = self.train_data.reset_index(drop=True)
        self.valid_data = self.valid_data.reset_index(drop=True)
        self.test_data = self.test_data.reset_index(drop=True)

        print self.train_data

        print self.train_data


    def generate_unary_transformations(self): # Fu,i
        unary_transformations = get_unary_transformations()

        applied_unary_transformations = []

        for attribute_i in range(self.number_attributes):
            for transformation_i in range(len(unary_transformations)):
                transformation = copy.deepcopy(unary_transformations[transformation_i])
                transformation.column_id = attribute_i
                applied_unary_transformations.append(transformation)


        applied_unary_transformations_output_dim = []

        for transformation_i in range(len(applied_unary_transformations)):
            self.transformer.transformers = [applied_unary_transformations[transformation_i]]

            #if transformer.transformers[0].output_space_size == None:
            self.transformer.fit()

            error = False
            try:
                transformed_data, target_data, feature_names = self.transformer.transform()
            except:
                error = True

            if not error:
                print self.train_data.columns
                new_train = pd.DataFrame(data=transformed_data[0], index=range(transformed_data[0].shape[0]),
                                         columns=feature_names)

                self.train_data = pd.concat([self.train_data, new_train], axis=1)

                new_test = pd.DataFrame(data=transformed_data[1], index=range(transformed_data[1].shape[0]),
                                         columns=feature_names)

                self.test_data = pd.concat([self.test_data, new_test], axis=1)

                new_valid = pd.DataFrame(data=transformed_data[2], index=range(transformed_data[2].shape[0]),
                                        columns=feature_names)

                self.valid_data = pd.concat([self.valid_data, new_valid], axis=1)

            #print applied_unary_transformations[transformation_i]
            #print self.test_data.columns
            #print self.train_data.columns
            #print self.train_data
            assert self.train_data.shape[1] == self.test_data.shape[1], "test != train: " + str(
                self.train_data.shape[1]) + " != " + str(self.test_data.shape[1])

            print "all: " + str(self.train_data.shape)



            applied_unary_transformations_output_dim.append(self.transformer.transformers[0].output_space_size)

            assert ((type(new_train) == type(None) and self.transformer.transformers[0].output_space_size==0)) or (new_train.shape[1] == self.transformer.transformers[0].output_space_size), str(
                self.transformer.transformers[0]) + ": " + str(new_train.shape[1]) + " != " + str(self.transformer.transformers[0].output_space_size)

        print str(applied_unary_transformations_output_dim)
        print str(np.sum(applied_unary_transformations_output_dim))
        self.unary_output_dimension_size = applied_unary_transformations_output_dim

        assert self.train_data.shape[1] == self.test_data.shape[1], "test != train: " + str(self.train_data.shape[1])  + " != " + str(self.test_data.shape[1])
        assert self.train_data.shape[1] == self.valid_data.shape[1], "valid != train"
        assert self.test_data.shape[1] == self.valid_data.shape[1], "valid != test"


    def generate_binary_transformations(self): # Fo,i
        binary_transformations = get_binary_transformations()

        applied_binary_transformations = []

        for attribute_a in range(self.train_data.shape[1]):
            for attribute_b in range(attribute_a+1, self.train_data.shape[1]):
                for transformation_i in range(len(binary_transformations)):
                    transformation = copy.deepcopy(binary_transformations[transformation_i])
                    transformation.column_a =attribute_a
                    transformation.column_b = attribute_b
                    applied_binary_transformations.append(transformation)

        print "hello"
        print len(applied_binary_transformations)

        for transformation_i in range(len(applied_binary_transformations)):
            current_transformation = applied_binary_transformations[transformation_i]

            train_data_column_a = self.train_data[self.train_data.columns[applied_binary_transformations[transformation_i].column_a]]
            train_data_column_b = self.train_data[self.train_data.columns[applied_binary_transformations[transformation_i].column_b]]

            test_data_column_a = self.test_data[
                self.test_data.columns[applied_binary_transformations[transformation_i].column_a]]
            test_data_column_b = self.test_data[
                self.test_data.columns[applied_binary_transformations[transformation_i].column_b]]

            valid_data_column_a = self.valid_data[
                self.valid_data.columns[applied_binary_transformations[transformation_i].column_a]]
            valid_data_column_b = self.valid_data[
                self.valid_data.columns[applied_binary_transformations[transformation_i].column_b]]

            current_transformation.fit1(train_data_column_a, train_data_column_b)

            transformed_train = current_transformation.transform1(train_data_column_a, train_data_column_b)
            transformed_test = current_transformation.transform1(test_data_column_a, test_data_column_b)
            transformed_valid = current_transformation.transform1(valid_data_column_a, valid_data_column_b)

            if type(self.train_data_bin) == type(None):
                self.train_data_bin = np.matrix(transformed_train).T #check if different
                self.test_data_bin = np.matrix(transformed_test).T
                self.valid_data_bin = np.matrix(transformed_valid).T
            else:
                self.train_data_bin = np.concatenate((self.train_data_bin, np.matrix(transformed_train).T), axis=1)
                self.test_data_bin = np.concatenate((self.test_data_bin, np.matrix(transformed_test).T), axis=1)
                self.valid_data_bin = np.concatenate((self.valid_data_bin, np.matrix(transformed_valid).T), axis=1)

                #print str(self.train_data_bin.shape) + " vs " + str(np.matrix(transformed_train).T.shape)

        print "binary size:" + str(self.train_data_bin.shape)


    def generate_binary_unary_transformations(self): # Fu,i
        unary_transformations = get_unary_transformations()

        applied_unary_transformations = []

        for attribute_i in range(self.train_data_bin.shape[1]):
            for transformation_i in range(len(unary_transformations)):
                transformation = copy.deepcopy(unary_transformations[transformation_i])
                transformation.column_id = attribute_i
                applied_unary_transformations.append(transformation)


        applied_unary_transformations_output_dim = []

        for transformation_i in range(len(applied_unary_transformations)):
            current_transformer = applied_unary_transformations[transformation_i]

            train_data_column = self.train_data_bin[:, applied_unary_transformations[transformation_i].column_id]
            test_data_column = self.test_data_bin[:, applied_unary_transformations[transformation_i].column_id]
            valid_data_column = self.valid_data_bin[:, applied_unary_transformations[transformation_i].column_id]

            current_transformer.fit1(train_data_column)

            error = False
            try:
                transformed_train = current_transformer.transform1(train_data_column)
                transformed_test = current_transformer.transform1(test_data_column)
                transformed_valid = current_transformer.transform1(valid_data_column)
            except:
                error = True

            if not error:
                if type(self.train_data_bin_una) == type(None):
                    self.train_data_bin_una = np.matrix(transformed_train)  # check if different
                    self.test_data_bin_una = np.matrix(transformed_test)
                    self.valid_data_bin_una = np.matrix(transformed_valid)
                else:
                    print str(self.train_data_bin_una.shape) + " vs " + str((np.matrix(transformed_train).T).shape)

                    try:
                        self.train_data_bin_una = np.concatenate((self.train_data_bin_una, np.matrix(transformed_train).T), axis=1)
                        self.test_data_bin_una = np.concatenate((self.test_data_bin_una, np.matrix(transformed_test).T), axis=1)
                        self.valid_data_bin_una = np.concatenate((self.valid_data_bin_una, np.matrix(transformed_valid).T), axis=1)
                    except:
                        self.train_data_bin_una = np.concatenate(
                            (self.train_data_bin_una, np.matrix(transformed_train)), axis=1)
                        self.test_data_bin_una = np.concatenate((self.test_data_bin_una, np.matrix(transformed_test)),
                                                                axis=1)
                        self.valid_data_bin_una = np.concatenate(
                            (self.valid_data_bin_una, np.matrix(transformed_valid)), axis=1)


        print self.train_data_bin_una.shape[1]


















if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    def remove_target_from_df(dataframe, target_id):
        dataframe.drop(dataframe.columns[target_id], axis=1, inplace=True)
        return dataframe

    dataset = ("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13)
    dataframe = pd.read_csv(dataset[0])

    target = np.array(dataframe[dataframe.columns[dataset[1]]])

    dataframe = remove_target_from_df(dataframe, dataset[1])

    g = Generator(dataframe, target)

    g.generate_unary_transformations()
    g.generate_binary_transformations()
    g.generate_binary_unary_transformations()


