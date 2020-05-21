import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.heatmap.heatmap_util import pivot2latex


data = pickle.load(open('/home/felix/phd/versions_dfs/pairs/current_heat_map_privacy_acc.pickle', "rb"))

accuracies = []
privacy = []
strategy= []
searchtime = []

map_real_id_to_unique = {}
map_unique_to_name = {}





for k,v in data.items():
	accuracies.append(round(k[0], 2))
	privacy.append(round(k[1], 3))
	searchtime.append(round(v[0], 2))
	strategy.append(v[1])

print(map_unique_to_name)

df = pd.DataFrame({'Minimum Accuracy': accuracies, 'Privacy Epsilon': privacy, 'Search time': searchtime, 'Fastest Strategy': strategy})

print(df)

ax = sns.heatmap(df.pivot("Minimum Accuracy", "Privacy Epsilon", "Search time"))
plt.show()

print(len(np.unique(strategy)))
print(np.unique(strategy))

#current_palette = sns.color_palette(palette='dark', n_colors=len(np.unique(strategy)))
current_palette = sns.color_palette(palette='colorblind', n_colors=len(np.unique(strategy)))
#current_palette = sns.palplot(sns.color_palette("hls",7))

#https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=7
flatui =['#e41a1c','#377eb8','#4daf4a','#984ea3']


fig, ax = plt.subplots(figsize=(3.5, 3.5))
my_pivot = df.pivot("Minimum Accuracy", "Privacy Epsilon", 'Fastest Strategy')
my_pivot.sort_index(level=0, ascending=False, inplace=True)
my_pivot.sort_index(axis=1, ascending=False, inplace=True)
print(my_pivot)


pivot2latex(my_pivot)
