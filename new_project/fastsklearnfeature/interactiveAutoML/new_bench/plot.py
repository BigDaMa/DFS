import pickle
import numpy as np
import pandas as pd

grid = pickle.load(open("/home/felix/phd/feature_constraints/bench/Kbest_grid.p", "rb"))

print(grid)

#complexity = grid['param_selection__k']
complexity = grid['param_selection__n_features_to_select']
accuracy = grid['mean_test_auc']
time = np.array(grid['mean_fit_time']) + np.array(grid['mean_score_time'])

step_time = np.zeros(len(time))
step_time[0] = time[0]
for i in range(1, len(time)):
	step_time[i] = step_time[i-1] + time[i]

print(complexity)
print(accuracy)
print(time)
print(step_time)

data = pd.DataFrame(data=np.array([complexity, accuracy, time]).transpose(), columns=['complexity', 'accuracy', 'time'])

data.to_csv('/home/felix/phd/feature_constraints/bench_experiments/RFE.csv', index=False)