from ml.kaggle.representation_learning.Transformer.Mapper.DateMapper import DateMapper
from ml.kaggle.representation_learning.Transformer.Mapper.LatitudeLongitudeMapper import LatitudeLongitudeMapper
import  kaggle.representation_learning.Transformer.Mapper.all as allmapper
import kaggle.representation_learning.Transformer.Mapper.categorical as catmapper
import kaggle.representation_learning.Transformer.Mapper.numeric as nummapper


from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import hstack



class Transformer:
    def __init__(self, dataset, target_column_id, number_clusters_for_target=30, seed=42, map=True):
        self.dataset = dataset
        self.transformers = []
        self.ids_parts = [[], [], []]
        self.target_column_id = target_column_id
        self.seed = seed

        self.y = np.array(dataset[dataset.columns[target_column_id]])

        # transform to classification problem
        if number_clusters_for_target > 1:
            newy = self.y.reshape(-1, 1)
            kmeans_labels = KMeans(n_clusters=number_clusters_for_target, random_state=self.seed).fit(newy)
            self.y = kmeans_labels.predict(newy)


        if map:
            self.map()




    def map(self):
        self.processed_columns = [self.target_column_id]
        self.mappers = [DateMapper(),
                        #
                        nummapper.BinarizeMapper(),
                        nummapper.BucketMapper(),
                        nummapper.ClusterDistMapper(),
                        nummapper.ClusterMapper(),
                        nummapper.CosMapper(),
                        nummapper.DegreesMapper(),
                        nummapper.DummyMapper(),
                        nummapper.IdentityMapper(),
                        nummapper.ImputerMapper(),
                        nummapper.LogMapper(),
                        nummapper.PlottingPositionMapper(),
                        nummapper.PolynomialMapper(),
                        nummapper.QuantileMapper(),
                        nummapper.RadiansMapper(),
                        nummapper.RankMapper(),
                        nummapper.RSHMapper(),
                        nummapper.ScaleMapper(),
                        nummapper.SigmoidMapper(),
                        nummapper.SinMapper(),
                        nummapper.SqrtMapper(),
                        nummapper.SquareMapper(),
                        nummapper.TanMapper(),
                        nummapper.ToIntMapper(),
                        nummapper.TrimTailMapper(),
                        nummapper.WinsorizeMapper(),
                        nummapper.ZscoreMapper(),

                        catmapper.StringOneHotMapper(),
                        catmapper.FrequencyMapper(),
                        catmapper.OrdinalMapper(),

                        allmapper.AvgWord2VecMapper(),
                        allmapper.HashingMapper(),
                        allmapper.LengthCountMapper(),
                        allmapper.NgramMapper(analyzer='char'),
                        allmapper.NgramMapper(analyzer='word'),
                        allmapper.ParseNumbersMapper(),

                        allmapper.SkipMapper()]
        self.attribute_position = np.zeros(self.dataset.shape[1], dtype=int)

        #mapping
        position_i = 0
        for mapper in self.mappers:
            self.processed_columns, self.attribute_position, self.transformers = mapper.map(self.dataset, self.processed_columns, self.attribute_position, position_i, self.transformers)
            position_i += 1



    def create_train_test(self, test_fraction=0.2):
        #X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.y, test_size=test_fraction, random_state=self.seed, stratify=self.y)
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_fraction, random_state=self.seed, stratify=y_train)

        X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.y, test_size=test_fraction, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_fraction, random_state=self.seed)

        self.ids_parts[0] = X_train.index.values
        self.ids_parts[1] = X_val.index.values
        self.ids_parts[2] = X_test.index.values

    def fit(self):
        if len(self.ids_parts[0]) == 0:
            self.create_train_test()

        for transformer_x in self.transformers:
            transformer_x.fit(self.dataset, self.ids_parts[0])

    def transform(self):
        transformed_data = [None] * 3
        target_data = [None] * 3

        for transformer_x in self.transformers:
            for part_dataset in range(3):
                if transformer_x.applicable:
                    transform_result = transformer_x.transform(self.dataset, self.ids_parts[part_dataset])
                    if type(transform_result) != type(None): #check for empty transformation
                        if type(transformed_data[part_dataset]) == type(None):
                            print str(transformer_x.__class__.__name__)
                            transformed_data[part_dataset] = transform_result
                        else:
                            print str(transformer_x.__class__.__name__)
                            print str(type(transformed_data[part_dataset])) + " : " + str(type(transform_result))
                            print str(transformed_data[part_dataset].shape) + " : " + str(transform_result.shape)
                            if "numpy" in str(type(transformed_data[part_dataset])) and "numpy" in str(type(transform_result)):
                                transformed_data[part_dataset] = np.hstack((transformed_data[part_dataset], transform_result))
                            else:
                                transformed_data[part_dataset] = hstack((transformed_data[part_dataset], transform_result)).tocsr()

        for part_dataset in range(3):
            target_data[part_dataset] = self.y[self.ids_parts[part_dataset]]


        feature_names = []

        for transformer_x in self.transformers:
            names = transformer_x.get_feature_names(self.dataset)
            if type(names) != type(None):
                feature_names.extend(names)

        return transformed_data, target_data, feature_names

    def next_transformation_for_attribute(self, column_id):
        new_transformers = []
        new_processed_columns = []
        #get corresponding current transformer
        for transformerx in self.transformers:
            t_columns = transformerx.get_involved_columns()
            if column_id in t_columns:
                for col in t_columns: #find next transformer for these columns
                    for position_i in range(self.attribute_position[col]+1, len(self.mappers)): #something wromg here
                        _, _, transformers_temp = self.mappers[position_i].map(self.dataset, new_processed_columns, np.copy(self.attribute_position), position_i, [])

                        for transformertemp_i in transformers_temp:
                            if col in transformertemp_i.get_involved_columns():
                                new_transformers.append(transformertemp_i)
                                new_processed_columns.extend(transformertemp_i.get_involved_columns())
                                self.attribute_position[col] = position_i
                                break
                        if col in new_processed_columns:
                            break

                        position_i += 1
            else:
                new_transformers.append(transformerx)
                new_processed_columns.extend(t_columns)

        self.transformers = new_transformers
        self.processed_columns = new_processed_columns

    def print_config(self):
        for transformerx in self.transformers:
            print "Transformer: " + str(transformerx.__class__.__name__) + " -> " + str(transformerx.get_involved_columns())