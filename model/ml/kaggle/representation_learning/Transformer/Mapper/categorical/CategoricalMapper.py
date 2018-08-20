class CategoricalMapper():

    def __init__(self, transformer_class):
        self.threshold_unique_values = 0.05
        self.transformer = transformer_class

    def map(self, dataset, processed_columns, attribute_position, position_i, transformers):
        for column_i in range(len(dataset.dtypes)):
            if not column_i in processed_columns:
                count = len(dataset[dataset.columns[column_i]].unique())
                fraction = count / float(len(dataset))

                if fraction < self.threshold_unique_values:
                    transformers.append(self.transformer(column_i))
                    processed_columns.append(column_i)
                    attribute_position[column_i] = position_i

        return processed_columns, attribute_position, transformers