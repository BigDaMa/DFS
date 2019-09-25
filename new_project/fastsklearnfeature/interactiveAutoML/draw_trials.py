import pickle
from hyperopt import Trials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_min_curve(losses, test):
	min_losses = [losses[0]]
	test_equi = [test[0]]
	for l in np.arange(1, len(losses)):
		if min_losses[-1] > losses[l]:
			min_losses.append(losses[l])
			test_equi.append(test[l])
		else:
			min_losses.append(min_losses[-1])
			test_equi.append(test_equi[-1])
	return min_losses, test_equi

all_features_200: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new2.p", "rb"))

raw_first: Trials = pickle.load(open("/home/felix/phd/feature_constraints/experiment1/trials_new.p", "rb"))

losses = [r['loss'] for r in all_features_200.results]
losses_raw = [r['loss'] for r in raw_first.results]

test = [r['test'] for r in all_features_200.results]
test_raw = [r['test'] for r in raw_first.results]



my_dict = raw_first.vals
my_dict['class'] = [1]*200

del my_dict['C']

pd.plotting.parallel_coordinates(pd.DataFrame(my_dict), class_column='class')
plt.show()



min_losses1, test_equi1 = create_min_curve(losses, test)
min_losses2, test_equi2 = create_min_curve(losses_raw, test_raw)

plt.plot(range(len(losses)), min_losses1, color='red', label='200 trials all features')
plt.plot(range(len(losses)), min_losses2, color='blue', label='100 trials raw features -> 100 trails all features')
plt.xlabel('Trials')
plt.ylabel('Loss (1.0 - AUC)')

leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.show()


plt.plot(range(len(losses)), test_equi1, color='red', label='200 trials all features')
plt.plot(range(len(losses)), test_equi2, color='blue', label='100 trials raw features -> 100 trails all features')
plt.xlabel('Trials')
plt.ylabel('AUC on Test')

leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

plt.show()
