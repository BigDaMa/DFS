import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X_test = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/X_test.p", "rb"))
y_test = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/y_test.p", "rb"))

names = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/names.p", "rb"))

my_trials: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new3.p", "rb"))

pred_best = my_trials.best_trial['result']['pred_test']


are_predictions_correct = np.equal(np.array(y_test.values), np.array(pred_best))


print(pd.DataFrame(X_test, columns=names))

from PyQt5.QtWidgets import QWidget,QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem,QApplication
import pandas as pd

app = QApplication([])
win = QWidget()
scroll = QScrollArea()
layout = QVBoxLayout()
table = QTableWidget()
scroll.setWidget(table)
layout.addWidget(table)
win.setLayout(layout)


df = pd.DataFrame(X_test, columns=names)

## add column
df['Misprediction'] = are_predictions_correct

table.setColumnCount(len(df.columns))
table.setRowCount(len(df.index))

fnames = [str(n) for n in names]
fnames.append('Is correct prediction?')

table.setHorizontalHeaderLabels(fnames)


for i in range(len(df.index)):
    for j in range(len(df.columns)):
        table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

win.show()
app.exec_()
