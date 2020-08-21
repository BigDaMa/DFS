import openml

test_holdout_dataset_id = 40945

dataset_hold = openml.datasets.get_dataset(dataset_id=test_holdout_dataset_id)
X_hold, y_hold, categorical_indicator_hold, attribute_names_hold = dataset_hold.get_data(
    dataset_format="dataframe",
    target=dataset_hold.default_target_attribute
)

print(X_hold.values)
print(y_hold.values)