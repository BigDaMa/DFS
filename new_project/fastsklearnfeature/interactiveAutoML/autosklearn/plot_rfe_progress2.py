import numpy as np
import matplotlib.pyplot as plt
import pickle

features = []
test_auc = []



all_data = pickle.load(open("/tmp/features_info.p", "rb"))

names = pickle.load(open("/tmp/featurenames.p", "rb"))


#print(all_data)

feature_lists = []

for i in range(len(all_data)):
	features.append(len(all_data[i][0]))
	feature_lists.append(all_data[i][0])
	test_auc.append(all_data[i][1])


for i in range(len(all_data)):
	my_str = ''
	for f in feature_lists[i]:
		my_str += str(names[f]) + ', '
	print(my_str + ": " + str(test_auc[i]))



plt.plot(features, test_auc, label='Our Method')

plt.hlines(y=0.7891584766584766, xmin=min(features), xmax=max(features),color='blue', label='Base Features')

plt.hlines(y=0.7986793611793612, xmin=min(features), xmax=max(features),color='red', label='1h AutoSklearn Ensemble')

plt.legend(loc='lower right')
plt.ylabel('Test Auc')
plt.xlabel('Number of Features')

plt.show()