from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.BucketTransformer import BucketTransformer
from kaggle.representation_learning.Transformer.Mapper.numeric.NumericMapper import NumericMapper

class BucketMapper(NumericMapper):

    def __init__(self, number_bins=10, qbucket=False):
        NumericMapper.__init__(self, BucketTransformer, {'number_bins': number_bins, 'qbucket': qbucket})
        self.number_bins = number_bins
        self.qbucket = qbucket

    def map(self, dataset, processed_columns, attribute_position, position_i, transformers):
        for column_i in range(len(dataset.dtypes)):
            if (dataset.dtypes[column_i] == 'int64' or dataset.dtypes[column_i] == 'float64') and not column_i in processed_columns:
                if len(dataset[dataset.columns[column_i]].unique()) > self.number_bins:
                    transformers.append(self.transformer(column_i, self.number_bins, self.qbucket))
                    processed_columns.append(column_i)
                    attribute_position[column_i] = position_i

        return processed_columns, attribute_position, transformers
