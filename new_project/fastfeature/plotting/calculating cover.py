import pickle
from fastfeature.plotting.plotter import cool_plotting

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_sorted_by_complexity_fold20_hyper_opt_1045min.p'

#heart
#file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
#my_range = (0.72, 0.88)
# heart also raw features
file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/chart.p'
my_range = (0.50, 0.88)



#diabetes
#file = '/home/felix/phd/fastfeatures/results/diabetes/chart.p'
#my_range = (0.72, 0.78)

all_data = pickle.load(open(file, "rb"))

interpretability = all_data['interpretability']
scores = all_data['new_scores']

#calculate cover

#get first local optima



print(interpretability)

import numpy as np
from scipy.signal import argrelextrema


ids = np.argsort(np.array(interpretability) * -1.0)
print("sorted " + str(interpretability[ids[0]]))

sorted_scores = np.array(scores)[ids]

cummulative = []
cur_max =-10
for i in range(len(sorted_scores)):
    if cur_max < sorted_scores[i]:
        cur_max = sorted_scores[i]
    cummulative.append(cur_max)

cum_flip = cummulative #np.flip(np.array(cummulative))


import matplotlib.pyplot as plt

plt.plot(np.array(interpretability)[ids], sorted_scores)
plt.plot(np.array(interpretability)[ids], cum_flip)

#plt.axvline(x=local[-1], color='r')

plt.show()