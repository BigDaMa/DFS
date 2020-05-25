import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import copy

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.heatmap.heatmap_util import pivot2latex

data = pickle.load(open('/home/felix/phd/versions_dfs/new_pairs/current_heat_map_safety_acc.pickle', "rb"))



accuracies = []
safety = []
strategy= []
searchtime = []


data_dict = copy.deepcopy(data)

for k,v in data.items():
	if k[1] < 0.75:
		del data_dict[k]
data = data_dict


for k,v in data.items():
	accuracies.append(round(k[0], 2))
	safety.append(round(k[1], 2))
	searchtime.append(round(v[0], 2))
	strategy.append(v[1])


df = pd.DataFrame({'Minimum Accuracy': accuracies, 'Minimum Safety': safety, 'Search time': searchtime, 'Fastest Strategy': strategy})


ax = sns.heatmap(df.pivot("Minimum Accuracy", "Minimum Safety", "Search time"))
plt.show()

print(len(np.unique(strategy)))
print(np.unique(strategy))

#current_palette = sns.color_palette(palette='dark', n_colors=len(np.unique(strategy)))
current_palette = sns.color_palette(palette='colorblind', n_colors=len(np.unique(strategy)))
#current_palette = sns.palplot(sns.color_palette("hls",7))

#https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=7
flatui =['#e41a1c','#377eb8','#4daf4a','#984ea3']


fig, ax = plt.subplots(figsize=(3.5, 3.5))
my_pivot = df.pivot("Minimum Accuracy", "Minimum Safety", 'Fastest Strategy')
my_pivot.sort_index(level=0, ascending=False, inplace=True)
#my_pivot.sort_index(axis=1, ascending=False, inplace=True)
print(my_pivot)


pivot2latex(my_pivot)
