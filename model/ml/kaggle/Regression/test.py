import category_encoders as ce

train = ['Brunswick East', 'Fitzroy', 'Williamstown', 'Newport', 'Balwyn North', 'Doncaster', 'Melbourne', 'Albert Park', 'Bentleigh', 'Northcote']
test = ['Fitzroy North', 'Fitzroy', 'Richmond', 'Surrey Hills', 'Blackburn', 'Port Melbourne', 'Footscray', 'Yarraville', 'Carnegie', 'Surrey Hills']

encoder = ce.HelmertEncoder()
encoder.fit(train)

train_t = encoder.transform(train)
test_t = encoder.transform(test)



print train_t.shape
print test_t.shape

import numpy as np
print np.square([2,3])

print np.log(0)
print "test"


list = [7, 6, 5, 7, 6, 7, 6, 6, 6, 4, 5, 6]
winner = np.argmax(list)
print winner