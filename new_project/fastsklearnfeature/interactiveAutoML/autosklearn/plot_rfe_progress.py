import numpy as np
import matplotlib.pyplot as plt

features = []
test_auc = []

with open("/home/felix/phd/experiment_f/german_credit") as file_in:
	for line in file_in:
		tokens = line.split(' ')
		features.append(int(tokens[2]))
		test_auc.append(float(tokens[-1]))


plt.plot(features, test_auc, label='Our Method')

plt.hlines(y=0.7891584766584766, xmin=min(features), xmax=max(features),color='blue', label='Base Features')

plt.hlines(y=0.7986793611793612, xmin=min(features), xmax=max(features),color='red', label='1h AutoSklearn Ensemble')

plt.legend(loc='lower right')
plt.ylabel('Test Auc')
plt.xlabel('Number of Features')

plt.show()