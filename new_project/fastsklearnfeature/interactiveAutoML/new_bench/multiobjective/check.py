import pickle

one_big_object = pickle.load(open('/tmp/metalearning_data.pickle', 'rb'))

print(one_big_object.keys())
print(one_big_object['best_strategy'])