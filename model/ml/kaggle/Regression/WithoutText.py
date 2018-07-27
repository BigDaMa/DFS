import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

import dateutil.parser
import time
import datetime
from sklearn import preprocessing
import xgboost as xgb
from scipy.sparse import hstack
from scipy import sparse


from math import radians, cos, sin
from sklearn.cluster import KMeans

from xgboost import plot_importance
from matplotlib import pyplot


#Latitude and Longitude to cartesian cordindates converesion
# Assuming Earth as sphere not ellipsoid
def to_cartesian_coordinates(lat, lon):
    lat = radians(lat)
    lon = radians(lon)

    R = 6371.230
    x = R * cos(lat) * cos(lon)
    y = R * cos(lat) * sin(lon)
    z = R * sin(lat)

    return (x, y, z)

def latlon2coordinates(lat, lon):
    coordinates_matrix = np.zeros((len(lat), 3))

    for latlon_i in range(len(lat)):
        coordinates_matrix[latlon_i] = to_cartesian_coordinates(lat[latlon_i], lon[latlon_i])

    return coordinates_matrix


def calc_distance_to_clusters(kmeans, coordinates_train):
    distance_matrix = np.zeros((len(coordinates_train), len(kmeans.cluster_centers_)))
    for cluster_i in range(len(kmeans.cluster_centers_)):
        for data_i in range(coordinates_train.shape[0]):
            distance_matrix[data_i, cluster_i] = np.linalg.norm(
                coordinates_train[data_i] - kmeans.cluster_centers_[cluster_i])
    return distance_matrix


def date_expansion(timestamps):
    date_expansion_matrix = np.zeros((len(timestamps[0]), len(timestamps)*2))
    for timestamp_i in range(len(timestamps)):
        for record_i in range(len(timestamps[timestamp_i])):
            my_date = datetime.datetime.fromtimestamp(timestamps[timestamp_i][record_i])
            date_expansion_matrix[record_i, 0] = my_date.weekday()
            date_expansion_matrix[record_i, 1] = my_date.month


    return date_expansion_matrix


use_date_conversion = True
use_one_hot = True
use_text_features = True
use_lat_long_conversion = True
use_cluster_lat_lon = True



with open('/home/felix/FastFeatures/kaggle/schema_2.csv') as f: #housing
#with open('/home/felix/FastFeatures/kaggle/schema_imdb.csv') as f: #housing
    for line in f:
        tokens = line.split("#")
        user = tokens[0]
        project = tokens[1]
        csv_file = tokens[2]
        type_var = tokens[3]
        task = tokens[4]
        target_column = int(tokens[5])

        mypath = "/home/felix/.kaggle/datasets/" + str(user) + "/" + str(project) + "/" + csv_file

        pandas_table = pd.read_csv(mypath, encoding="utf-8", parse_dates=True)

        target = pandas_table[pandas_table.columns[target_column]]
        processed_colums = [target_column]


        date_ids = []
        timestamps = []

        #date conversion
        for column_i in range(len(pandas_table.columns)):
            if pandas_table.dtypes[column_i] == 'object' and not column_i in processed_colums:
                try:
                    yourdate = dateutil.parser.parse(pandas_table.values[0, column_i])
                    print yourdate
                    print time.mktime((yourdate).timetuple())

                    timestamps.append(pandas_table[pandas_table.columns[column_i]].apply(lambda x: time.mktime((dateutil.parser.parse(x)).timetuple())).values)
                    date_ids.append(column_i)
                except ValueError:
                    continue
                except TypeError:
                    continue

        print "date ids: " + str(date_ids)
        if len(date_ids) > 0:
            processed_colums.extend(date_ids)

        #one hot encoding

        one_hot_matrix = []
        one_hot_columns = []
        one_hot_featurenames = []

        for column_i in range(len(pandas_table.dtypes)):
            if (pandas_table.dtypes[column_i] == 'object' or \
                    'post' in pandas_table.columns[column_i].lower() or 'plz' in pandas_table.columns[column_i].lower() \
                ) \
                    and not column_i in processed_colums:
                count = len(pandas_table[pandas_table.columns[column_i]].unique())
                fraction = count/float(len(pandas_table))

                if fraction < 0.05:
                    print "one hot " + str(pandas_table.columns[column_i]) + ": " + str(count) + " -> " + str(fraction)
                    one_hot_columns.append(column_i)

                    pandas_table[pandas_table.columns[column_i]] = pandas_table[pandas_table.columns[column_i]].apply(
                        lambda x: str(x))

                    column_data = pandas_table.values[:, column_i]
                    lb = preprocessing.LabelBinarizer()
                    onehot_rep = np.matrix(lb.fit_transform(column_data))

                    internal_names = []
                    if len(lb.classes_) > 2:
                        for class_i in range(len(lb.classes_)):
                            internal_names.append(str(pandas_table.columns[column_i]) + ":" + str(lb.classes_[class_i]))
                    else:
                        internal_names = [str(pandas_table.columns[column_i]) + "one_hot_enc"]

                    one_hot_featurenames.extend(internal_names)

                    if len(one_hot_matrix) == 0:
                        one_hot_matrix = onehot_rep
                    else:
                        print one_hot_matrix.shape
                        print onehot_rep.shape
                        one_hot_matrix = np.hstack((one_hot_matrix, onehot_rep))

                    assert onehot_rep.shape[1] == len(internal_names), "one hot problem: " + str(internal_names) + "_> " + str(onehot_rep.shape[1])
                else:
                    print "no one hot " + str(pandas_table.columns[column_i]) + ": " + str(count) + " -> " + str(fraction)

        processed_colums.extend(one_hot_columns)


        #detect latitude and longitude
        # convert to spherical coordinates
        numeric_features = []
        if use_lat_long_conversion:
            lat_id = -1
            lon_id = -1
            for column_i in range(len(pandas_table.dtypes)):
                if pandas_table.dtypes[column_i] == 'float64' and not column_i in processed_colums:
                    min_value = pandas_table[pandas_table.columns[column_i]].min(axis=0)
                    max_value = pandas_table[pandas_table.columns[column_i]].max(axis=0)
                    if min_value >= -180.0 and max_value <= 180.0:
                        print pandas_table.columns[column_i]
                        if "lat" in pandas_table.columns[column_i].lower():
                            lat_id = column_i
                        if "lon" in pandas_table.columns[column_i].lower():
                            lon_id = column_i

            print "ids: " + str(lat_id) + " - " + str(lon_id)

            coordinates = []
            if lat_id != -1 and lon_id != -1:
                lat_col = pandas_table[pandas_table.columns[lat_id]].values
                lon_col = pandas_table[pandas_table.columns[lon_id]].values

                coordinates = latlon2coordinates(lat_col, lon_col)

                print "lat and longitude converted"

                processed_colums.extend([lat_id, lon_id])


        numeric_features = []
        feature_names = []
        for column_i in range(len(pandas_table.dtypes)):
            if (pandas_table.dtypes[column_i] == 'int64' or pandas_table.dtypes[column_i] == 'float64') and not column_i in processed_colums:
                numeric_features.append(column_i)
                feature_names.append(pandas_table.columns[column_i] + "_numeric")

        text_features = []
        for column_i in range(len(pandas_table.dtypes)):
            if (pandas_table.dtypes[column_i] == 'object') and not column_i in processed_colums:
                text_features.append(column_i)



        processed_colums.extend(numeric_features)


        print np.sort(processed_colums)

        #print "sum: " + str(one_hot_matrix.shape[1] + len(numeric_features))

        #not processed features:
        print "not processed features: " + str(len(pandas_table.columns) - len(processed_colums))

        pandas_table = pandas_table.fillna('0.0')



        print pandas_table.dtypes

        matrix = pandas_table.values
        X = matrix[:, numeric_features]
        y = np.array(matrix[:, target_column])

        #hstack onehot encoded features
        if use_one_hot and len(one_hot_matrix) > 0:
            X = np.hstack((X, one_hot_matrix))
            feature_names.extend(one_hot_featurenames)

        if use_date_conversion and len(timestamps) > 0:
            for timestamp_i in range(len(timestamps)):
                X = np.hstack((X, np.matrix(timestamps[timestamp_i]).T))
                feature_names.append(pandas_table.columns[date_ids[timestamp_i]] + "_timestamp")

            '''
            # date expansion            
            X = np.hstack((X, date_expansion(timestamps)))

            for date_id in date_ids:
                feature_names.append(pandas_table.columns[date_id] + "_weekday")
                feature_names.append(pandas_table.columns[date_id] + "_month")
            '''



        #clustering features
        if use_lat_long_conversion and len(coordinates) > 0:
            X = np.hstack((X, coordinates))
            feature_names.extend(['coordinate_x', 'coordinate_y', 'coordinate_z'])

        X_train_pd, X_test_pd, y_train, y_test = train_test_split(pandas_table, y, test_size=0.33, random_state=42)

        print X_train_pd.index.values
        X_train = X[X_train_pd.index.values, :]
        X_test = X[X_test_pd.index.values, :]

        if use_cluster_lat_lon and len(coordinates) > 0:
            coordinates_train = X_train[:, (X_train.shape[1]-3): X_train.shape[1]]
            coordinates_test = X_test[:, (X_test.shape[1] - 3): X_test.shape[1]]

            number_clusters = 10
            kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(coordinates_train)

            X_train = np.hstack((X_train, calc_distance_to_clusters(kmeans, coordinates_train)))
            X_test = np.hstack((X_test, calc_distance_to_clusters(kmeans, coordinates_test)))

            for cluster_i in range(number_clusters):
                feature_names.append("Cluster distance " + str(cluster_i))


        #bag of words
        if use_text_features and len(text_features) > 0:
            pipeline = Pipeline([
                ('vect', CountVectorizer(analyzer='char')),
                ('tfidf', TfidfTransformer()),
            ])

            for text_col_i in range(len(text_features)):
                text_col_train = pandas_table[pandas_table.columns[text_features[text_col_i]]].values[X_train_pd.index.values]
                text_col_test = pandas_table[pandas_table.columns[text_features[text_col_i]]].values[X_test_pd.index.values]
                ngram_rep_train = sparse.csr_matrix(pipeline.fit_transform(text_col_train), dtype=np.float64)
                ngram_rep_test = sparse.csr_matrix(pipeline.transform(text_col_test), dtype=np.float64)

                print type(X_train).__name__

                if type(X_train).__name__ == 'matrix':
                    sparsified_train = sparse.csr_matrix(np.matrix(X_train, dtype='float64'))
                    sparsified_test = sparse.csr_matrix(np.matrix(X_test, dtype='float64'))
                else:
                    sparsified_train = X_train
                    sparsified_test = X_test

                for word_i in range(len(pipeline.named_steps['vect'].vocabulary_)):
                    featurenames_vect = pipeline.named_steps['vect'].get_feature_names()
                    feature_names.append(pandas_table.columns[text_features[text_col_i]] + "_" + featurenames_vect[word_i] + "_text")

                X_train = hstack((sparsified_train, ngram_rep_train)).tocsr()
                X_test = hstack((sparsified_test, ngram_rep_test)).tocsr()



        #regr = linear_model.LinearRegression()
        #regr = linear_model.Ridge(alpha=.5)

        #quadratic_featurizer = PolynomialFeatures(degree=3)
        #X_train = quadratic_featurizer.fit_transform(X_train)
        #X_test = quadratic_featurizer.transform(X_test)

        #regr = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression())])

        #print X_train

        print y_train

        regr = xgb.XGBRegressor(nthread=4)
        regr.fit(X_train, y_train)


        #get feature importance
        b = regr.get_booster()
        fs = b.get_score('', importance_type='gain')
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        sorted = np.argsort(-all_features)


        print len(all_features)
        print len(feature_names)
        print list(feature_names)
        assert len(all_features) == len(feature_names), "Oh no! This assertion failed!"


        # Visualize model
        number_of_features = 10
        fig, ax = plt.subplots()
        show_features = np.array(feature_names)[sorted][0:number_of_features]
        y_pos = np.arange(len(show_features))
        performance = all_features[sorted][0:number_of_features]
        ax.barh(y_pos, performance, align='center',
                color='green', ecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(show_features)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Gain')
        plt.show()

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The mean squared error
        print("Mean squared error: %.8f" % mean_squared_error(y_test, y_pred))

        rms = sqrt(mean_squared_error(y_test, y_pred))
        print "RMSE: " + str(rms)

        print('Variance score: %.8f' % r2_score(y_test, y_pred))

        ids = np.argsort(y_test)

        plt.plot(range(len(y_test)), y_pred[ids], color='red', linewidth=3)
        plt.plot(range(len(y_test)), y_test[ids], color='blue', linewidth=3)

        plt.show()


        break