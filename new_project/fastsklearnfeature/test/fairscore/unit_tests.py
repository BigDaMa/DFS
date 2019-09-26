import numpy as np
from fastsklearnfeature.test.fairscore.fair_measure import true_positive_rate_score
import pandas as pd

y_true = ['good', 'bad', 'good', 'bad']
y_predict = np.array(['good', 'bad', 'bad', 'bad'])
sensitive = np.array(['m', 'm', 'f', 'f'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 0.5


y_true = ['good', 'bad', 'good', 'bad']
y_predict = np.array(['good', 'bad', 'good', 'bad'])
sensitive = np.array(['m', 'm', 'f', 'f'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 0.0

y_true = ['good', 'bad', 'good', 'bad']
y_predict = np.array(['good', 'good', 'good', 'good'])
sensitive = np.array(['m', 'm', 'f', 'f'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 0.0

y_true = ['good', 'bad', 'good', 'bad']
y_predict = np.array(['bad', 'bad', 'bad', 'bad'])
sensitive = np.array(['m', 'm', 'f', 'f'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 0.0

y_true = ['good', 'bad', 'good', 'bad']
y_predict = np.array(['good', 'good', 'bad', 'bad'])
sensitive = np.array(['m', 'm', 'm', 'm'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 0.0


y_true = ['good', 'bad', 'good', 'bad']
y_predict = np.array(['bad', 'bad', 'good', 'bad'])
sensitive = np.array(['m', 'm', 'f', 'f'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 0.5



y_true = ['good', 'bad', 'bad', 'good', 'bad', 'bad']
y_predict = np.array(['bad', 'bad', 'bad', 'good', 'bad', 'bad'])
sensitive = np.array(['m', 'm', 'm', 'f', 'f', 'f'])

assert true_positive_rate_score(pd.DataFrame(y_true), y_predict, sensitive, labels=['bad', 'good']) == 1.0/3

