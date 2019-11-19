import matplotlib.pyplot as plt
import pickle



import glob

file_lists = glob.glob("/tmp/all*.p")

for f_name in file_lists:
	map_k_to_results = pickle.load(open(f_name, "rb"))

	ks = []
	runtimes = []
	accuracies = []

	for k,value in map_k_to_results.items():
		ks.append(k)
		runtimes.append(value[1])
		accuracies.append(value[0])

	plt.plot(ks, accuracies, label=f_name[8:])

plt.legend()
plt.show()