import matplotlib.pyplot as plt
import pickle



import glob


path = '/tmp/'
#path = '/home/felix/phd/feature_constraints/bestk_promoters/'
#path = '/home/felix/phd/feature_constraints/bestk_experiments_madelon/'
#path = '/home/felix/phd/feature_constraints/bestk_arcene/'
#path = '/home/felix/phd/feature_constraints/bestk_tumor/'


#path = '/home/felix/phd/feature_constraints/data_promoters_knn/'
#path = '/home/felix/phd/feature_constraints/data_promoters_logistic_regression/'
#path = '/home/felix/phd/feature_constraints/data_promoters_dec_tree/'

#path = '/home/felix/phd/feature_constraints/madelon_knn/'
#path = '/home/felix/phd/feature_constraints/cancer_knn/'

file_lists = glob.glob(path + "all*.p")

for f_name in file_lists:
	map_k_to_results = pickle.load(open(f_name, "rb"))

	ks = []
	runtimes = []
	accuracies = []

	for k,value in map_k_to_results.items():
		if k>= 0:
			ks.append(k)
			runtimes.append(value[1])
			#accuracies.append(value[0]) #cv score
			accuracies.append(value[2])  # test score


	plt.plot(ks, accuracies, label=f_name.split('/')[-1][3:].split('.')[0])

plt.legend(loc=(1.04,0))
plt.xlabel('Number of Features')
plt.ylabel('AUC')
plt.subplots_adjust(right=0.7)
plt.show()