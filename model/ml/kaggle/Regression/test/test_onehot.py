from sklearn import preprocessing

data = ['h','h','h','u','h','h','u','h','h','h']
new_data = ['u','h','u','h','t','u','h','h','h','h']

one_hot_model = preprocessing.LabelBinarizer()

one_hot_model.fit(data)
print one_hot_model.transform(data)
print one_hot_model.transform(new_data)