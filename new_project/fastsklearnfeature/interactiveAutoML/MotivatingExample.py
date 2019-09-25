import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import r2_score
import eli5
from yellowbrick.features import rank2d
import matplotlib.pyplot as plt

train = pd.read_csv('/home/felix/phd/interactiveAutoML/train.csv')
test = pd.read_csv('/home/felix/phd/interactiveAutoML/test.csv')


train_y = train.SalePrice
train_X = train[train.columns.difference(['SalePrice', 'Id'])].values

preprocessing = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
	                 ('onehot', OneHotEncoder())])


#visualizer = rank2d(preprocessing.fit_transform(train_X).todense(),  train_y)
#plt.show()

y_test = train.SalePrice
X_text = train[train.columns.difference(['SalePrice', 'Id'])].values


pipeline = Pipeline([('preprocessing', preprocessing),
					 ('tree', DecisionTreeRegressor(criterion='mse', random_state=1,max_leaf_nodes=100))])


# Fit Model
pipeline.fit(train_X, train_y)

predictions = pipeline.predict(X_text)
print(r2_score(y_test, predictions))


import eli5
print(eli5.format_as_text(eli5.explain_weights(pipeline.named_steps['tree'])))