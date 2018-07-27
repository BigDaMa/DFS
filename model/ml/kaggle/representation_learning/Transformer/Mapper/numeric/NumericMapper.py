from kaggle.representation_learning.Transformer.TransformerImplementations.numeric.LogTransformer import LogTransformer

class NumericMapper():

    def __init__(self, transformer_class):
        self.transformer = transformer_class


    def map(self, dataset, processed_columns, attribute_position, position_i, transformers):
        for column_i in range(len(dataset.dtypes)):
            if (dataset.dtypes[column_i] == 'int64' or dataset.dtypes[column_i] == 'float64') and not column_i in processed_columns:
                transformers.append(self.transformer(column_i))
                processed_columns.append(column_i)
                attribute_position[column_i] = position_i

        return processed_columns, attribute_position, transformers
