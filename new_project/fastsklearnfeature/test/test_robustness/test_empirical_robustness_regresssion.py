from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from fastsklearnfeature.test.test_robustness.FGM_Regression import FastGradientMethod
from fastsklearnfeature.test.test_robustness.LinearRegressionSKlearn import ScikitlearnLinearRegression
from sklearn.metrics import r2_score
import numpy.linalg as la

diabetes = load_boston()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

regressor = ScikitlearnLinearRegression(model=model)
attack = FastGradientMethod(estimator=regressor, eps=0.1, batch_size=1)
X_test_attacked = attack.generate(X_test, y_test)

y_test_attacked = model.predict(X_test_attacked)
y_test_pred = model.predict(X_test)

print('r2 score on original data: ' + str(r2_score(y_true=y_test, y_pred=y_test_pred)))
print('r2 score on corrupted data: ' + str(r2_score(y_true=y_test, y_pred=y_test_attacked)))