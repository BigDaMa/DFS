import category_encoders as ce
import numpy as np
import pandas as pd

train = ['Brunswick East', 'Fitzroy', 'Williamstown', 'Newport', 'Balwyn North', 'Doncaster', 'Melbourne', 'Albert Park', 'Bentleigh', 'Northcote']
test = ['Fitzroy North', 'Fitzroy', 'Richmond', 'Surrey Hills', 'Blackburn', 'Port Melbourne', 'Footscray', 'Yarraville', 'Carnegie', 'Surrey Hills']

train_pd = pd.DataFrame(np.matrix(train).T)
test_pd = pd.DataFrame(np.matrix(test).T)

encoder = ce.HelmertEncoder()
encoder.fit(train_pd)

train_t = encoder.transform(train_pd)
test_t = encoder.transform(test_pd)

print train_t.shape
print test_t.shape

'''
import numpy as np
print np.square([2,3])

print np.log(0)
print "test"


list = [7, 6, 5, 7, 6, 7, 6, 6, 6, 4, 5, 6]
winner = np.argmax(list)
print winner


def str_length(mystring):
    return len(str(mystring))

str_length = np.vectorize(str_length, otypes=[np.int])

test_array = np.array(['hallo', 'test', '12'])

print str_length(test_array)
'''