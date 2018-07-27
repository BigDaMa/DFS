from kaggle.representation_learning.Transformer.TransformerImplementations.DateTransformer import DateTransformer
import dateutil.parser


class DateMapper():

    def __init__(self):
        return


    def map(self, dataset, processed_columns, attribute_position, position_i, transformers):
        for column_i in range(len(dataset.columns)):
            if dataset.dtypes[column_i] == 'object' and not column_i in processed_columns:
                try:
                    yourdate = dateutil.parser.parse(dataset.values[0, column_i])
                    transformers.append(DateTransformer(column_i))
                    processed_columns.append(column_i)
                    attribute_position[column_i] = position_i
                except ValueError:
                    continue
                except TypeError:
                    continue

        return processed_columns, attribute_position, transformers
