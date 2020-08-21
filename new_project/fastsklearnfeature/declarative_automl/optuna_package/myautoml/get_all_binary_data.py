import openml
import sklearn
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels

def get_data(data_id, randomstate=42):
    dataset = openml.datasets.get_dataset(dataset_id=data_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )



    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                y,
                                                                                random_state=randomstate,
                                                                                stratify=y,
                                                                                train_size=0.6)

    calculate_all_metafeatures_with_labels(X_train, y_train, categorical=categorical_indicator,
                                           dataset_name='data')

    #return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names

datasets = openml.datasets.list_datasets(number_classes='2')

print(len(datasets))

bin_datasets = []

for k,v in datasets.items():
    if v['version'] == 1 and not 'BNG(' in v['name'] and not 'FOREX' in v['name']:
        if k != 274:
            try:
                print(str(k) + ':' + str(v))
                get_data(k)
                bin_datasets.append(k)
            except:
                pass

print(len(bin_datasets))

print(bin_datasets)

