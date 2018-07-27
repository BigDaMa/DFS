import openml
from sklearn import preprocessing, tree, pipeline

# Set the OpenML API Key which is required to upload your runs.
# You can get your own API by signing up to OpenML.org.
openml.config.apikey = 'd1c17488c247f636c97303eb7d4b0402'


datasets = openml.datasets.list_datasets()

list_string_data = []

for data_id in datasets.iterkeys():
    data = openml.datasets.get_dataset(data_id)

    for feature in data.features.itervalues():
        if feature.data_type == 'string':
            list_string_data.append(data.dataset_id)
            print data.dataset_id
            break




'''
for feature in data.features.itervalues():
    if feature.data_type == 'string':
        print "yeah"
    else:
        print "booh"
'''

