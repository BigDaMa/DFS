import numpy as np
from math import radians, cos, sin

class LatitudeLongitudeTransformer():

    def __init__(self, latitude_id, longitude_id):
        self.latitude_id = latitude_id
        self.longitude_id = longitude_id


    def fit(self, dataset, ids):
        #nothing
        return

    def to_cartesian_coordinates(self, lat, lon):
        lat = radians(lat)
        lon = radians(lon)

        R = 6371.230
        x = R * cos(lat) * cos(lon)
        y = R * cos(lat) * sin(lon)
        z = R * sin(lat)

        return (x, y, z)

    def latlon2coordinates(self, lat, lon):
        coordinates_matrix = np.zeros((len(lat), 3))

        for latlon_i in range(len(lat)):
            coordinates_matrix[latlon_i] = self.to_cartesian_coordinates(lat[latlon_i], lon[latlon_i])

        return coordinates_matrix

    def transform(self, dataset, ids):
        lat_col = dataset[dataset.columns[self.latitude_id]].values[ids]
        lon_col = dataset[dataset.columns[self.longitude_id]].values[ids]

        coordinates = self.latlon2coordinates(lat_col, lon_col)

        return coordinates

    def get_feature_names(self, dataset):
        names = []

        names.append(str(self.latitude_id) + "," + str(self.longitude_id) + '#' + str(dataset.columns[self.latitude_id]) + "," + str(dataset.columns[self.longitude_id]) + "#" + "coordinate_x")
        names.append(str(self.latitude_id) + "," + str(self.longitude_id) + '#' + str(dataset.columns[self.latitude_id]) + "," + str(dataset.columns[self.longitude_id]) + "#" + "coordinate_y")
        names.append(str(self.latitude_id) + "," + str(self.longitude_id) + '#' + str(dataset.columns[self.latitude_id]) + "," + str(dataset.columns[self.longitude_id]) + "#" + "coordinate_z")

        return names

    def get_involved_columns(self):
        return [self.latitude_id, self.longitude_id]
