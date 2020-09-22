import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.heatmap.heatmap_util import pivot2latex
import pickle
import copy


data = pickle.load(open('/home/felix/phd2/heatmaps_pair_constraints/current_heat_map_fair_acc.pickle', "rb"))
#done

data_machine4 = {}
#data_machine4 = {(0.8400000000000001, 0.8): (19.320881652832032, 1), (0.8400000000000001, 0.811): (190.93613902727762, 8), (0.8400000000000001, 0.8220000000000001): (305.33650310834247, 8), (0.8400000000000001, 0.8330000000000001): (289.2884831428528, 9), (0.8400000000000001, 0.8440000000000001): (419.2489050626755, 8), (0.8400000000000001, 0.8550000000000001): (263.30750918388367, 8), (0.8400000000000001, 0.8660000000000001): (643.6740255355835, 9), (0.8600000000000001, 0.8): (7.100977277755737, 1), (0.8600000000000001, 0.811): (213.19025468826294, 8), (0.8600000000000001, 0.8220000000000001): (233.07127285003662, 9), (0.8600000000000001, 0.8330000000000001): (193.67269146442413, 8), (0.8600000000000001, 0.8440000000000001): (149.45733833312988, 8), (0.8800000000000001, 0.8): (12.070556211471558, 1), (0.8800000000000001, 0.811): (6.706651926040649, 1), (0.8800000000000001, 0.8220000000000001): (520.6932002703348, 8), (0.8800000000000001, 0.8330000000000001): (558.4049159288406, 9)}
#data = {**data , **data_machine4}


data_dict = copy.deepcopy(data)

for k,v in data.items():
	if k[1] > 0.89:
		del data_dict[k]
data = data_dict


accuracies = []
fairness = []
strategy= []
searchtime = []

map_real_id_to_unique = {}
map_unique_to_name = {}




for k,v in data.items():
	accuracies.append(round(k[0], 2))
	fairness.append(round(k[1], 2))
	searchtime.append(round(v[0], 2))
	strategy.append(v[1])

print(map_unique_to_name)

df = pd.DataFrame({'Minimum Accuracy': accuracies, 'Minimum Fairness': fairness, 'Search time': searchtime, 'Fastest Strategy': strategy})


#ax = sns.heatmap(df.pivot("Minimum Accuracy", "Minimum Fairness", "Search time"))
#plt.show()

print(len(np.unique(strategy)))
print(np.unique(strategy))

#current_palette = sns.color_palette(palette='dark', n_colors=len(np.unique(strategy)))
current_palette = sns.color_palette(palette='colorblind', n_colors=len(np.unique(strategy)))
#current_palette = sns.palplot(sns.color_palette("hls",7))

#https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=7
flatui =['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
#flatui=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
current_palette = sns.color_palette(flatui)



fig, ax = plt.subplots(figsize=(3.5, 3.5))
my_pivot = df.pivot("Minimum Accuracy", "Minimum Fairness", 'Fastest Strategy')
my_pivot.sort_index(level=0, ascending=False, inplace=True)
print(my_pivot)
sns.heatmap(my_pivot, cbar=False, cmap=current_palette, ax=ax)


pivot2latex(my_pivot)