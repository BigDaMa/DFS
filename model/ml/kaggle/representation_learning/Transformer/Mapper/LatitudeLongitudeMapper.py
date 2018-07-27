from kaggle.representation_learning.Transformer.TransformerImplementations.parser.LatitudeLongitudeTransformer import LatitudeLongitudeTransformer

class LatitudeLongitudeMapper():

    def __init__(self):
        return


    def map(self, dataset, processed_columns, attribute_position, position_i, transformers):
            lat_id = -1
            lon_id = -1
            for column_i in range(len(dataset.dtypes)):
                if dataset.dtypes[column_i] == 'float64' and not column_i in processed_columns:
                    min_value = dataset[dataset.columns[column_i]].min(axis=0)
                    max_value = dataset[dataset.columns[column_i]].max(axis=0)
                    if min_value >= -180.0 and max_value <= 180.0:
                        if "lat" in dataset.columns[column_i].lower():
                            lat_id = column_i
                        if "lon" in dataset.columns[column_i].lower():
                            lon_id = column_i

            if lat_id != -1 and lon_id != -1:
                transformers.append(LatitudeLongitudeTransformer(lat_id, lon_id))
                processed_columns.extend([lat_id, lon_id])
                attribute_position[lat_id] = position_i
                attribute_position[lon_id] = position_i

            return processed_columns, attribute_position, transformers