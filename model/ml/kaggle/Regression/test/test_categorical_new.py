import category_encoders as ce
import numpy as np
import pandas as pd
from patsy.contrasts import Helmert
from statsmodels.formula.api import ols

levels = [1,2,3,4]

contrast = Helmert().code_without_intercept(levels)

train =  np.array([1,2,3,4,5,6,3,3,3])
#train = ['Brunswick East', 'Fitzroy', 'Williamstown', 'Newport', 'Balwyn North', 'Doncaster', 'Melbourne', 'Albert Park', 'Bentleigh', 'Northcote']
#test = ['Fitzroy North', 'Fitzroy', 'Richmond', 'Surrey Hills', 'Blackburn', 'Port Melbourne', 'Footscray', 'Yarraville', 'Carnegie', 'Surrey Hills']

mod = ols("write ~ C(a, Helmert)", data=train)
res = mod.fit()
print res

'''
train_t = encoder.transform(train)

new_test = train
map_id_2_id = {}

for i in range(len(test)):
    if test[i] in train:
        map_id_2_id[len(new_test)] = i
        new_test.append(test[i])


new_test.extend(train)

test_t = encoder.transform(new_test)

print train_t.shape
print test_t.shape
'''