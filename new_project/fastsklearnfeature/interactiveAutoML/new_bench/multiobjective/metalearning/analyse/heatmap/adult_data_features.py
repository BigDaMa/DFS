import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.heatmap.heatmap_util import pivot2latex
import pickle

data = pickle.load(open('/home/felix/phd/versions_dfs/new_pairs/current_heat_map_complexity_acc.pickle', "rb"))

accuracies = []
max_number_features = []
strategy= []
searchtime = []

map_real_id_to_unique = {}
map_unique_to_name = {}




for k,v in data.items():
	if k[1] <= 1.0:
		accuracies.append(round(k[0], 2))
		max_number_features.append(round(k[1], 2))
		searchtime.append(round(v[0], 2))
		strategy.append(v[1])

print(map_unique_to_name)

df = pd.DataFrame({'Minimum Accuracy': accuracies, 'Maximum Features': max_number_features, 'Search time': searchtime, 'Fastest Strategy': strategy})


ax = sns.heatmap(df.pivot("Minimum Accuracy", "Maximum Features", "Search time"))
plt.show()

print(len(np.unique(strategy)))
print(np.unique(strategy))

#current_palette = sns.color_palette(palette='dark', n_colors=len(np.unique(strategy)))
current_palette = sns.color_palette(palette='colorblind', n_colors=len(np.unique(strategy)))
#current_palette = sns.palplot(sns.color_palette("hls",7))

#https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=7
flatui =['#e41a1c','#377eb8','#4daf4a','#984ea3']
current_palette = sns.color_palette(flatui)

#sns.palplot(current_palette)


fig, ax = plt.subplots(figsize=(3.5, 3.5))
my_pivot = df.pivot("Minimum Accuracy", "Maximum Features", 'Fastest Strategy')
my_pivot.sort_index(level=0, ascending=False, inplace=True)
my_pivot.sort_index(axis=1, ascending=False, inplace=True)
print(my_pivot)
sns.heatmap(my_pivot, cbar=False, cmap=current_palette, ax=ax)

print(my_pivot.values)
print(my_pivot.axes)




pivot2latex(my_pivot)