import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import roc_auc_score
import eli5
from yellowbrick.features import rank2d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('/home/felix/phd/interactiveAutoML/dataset_31_credit-g.csv')


y = data['class']
X = data[data.columns.difference(['class'])].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


preprocessing = Pipeline([('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
	                 ('onehot', OneHotEncoder())])


pipeline = Pipeline([('preprocessing', preprocessing),
					 ('tree', DecisionTreeClassifier())])


# Fit Model
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
print(roc_auc_score(y_test, predictions))


import eli5
print(eli5.format_as_text(eli5.explain_weights(pipeline.named_steps['tree'])))