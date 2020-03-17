import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib as mpl

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.heatmap.heatmap_util import pivot2latex


mappnames = {1:'var',
			 2: 'chi2',
			 3:'FCBF',
			 4: 'Fisher score',
			 5: 'mutual_info_classif',
			 6: 'MCFS',
			 7: 'ReliefF',
			 8: 'TPE',
             9: 'simulated_annealing',
			 10: 'NSGA-II',
			 11: 'exhaustive',
			 12: 'forward_selection',
			 13: 'backward_selection',
			 14: 'forward_floating_selection',
			 15: 'backward_floating_selection',
			 16: 'recursive_feature_elimination'
			 }



data = {(0.7, 10): (55.655747294425964, 1), (0.7, 7): (54.35564422607422, 1), (0.7, 3): (40.93184566497803, 1), (0.7, 1): (49.16838455200195, 1), (0.7, 0.7): (37.02667677402496, 1), (0.7, 0.3): (31.27279245853424, 1), (0.7, 0.1): (142.2492516040802, 1), (0.721, 10): (38.27897584438324, 1), (0.721, 7): (47.86325240135193, 1), (0.721, 3): (52.43798291683197, 1), (0.721, 1): (47.880462408065796, 1), (0.721, 0.7): (35.739949464797974, 1), (0.721, 0.3): (28.448721647262573, 1), (0.721, 0.1): (21.954465866088867, 1), (0.742, 10): (46.85345256328583, 1), (0.742, 7): (43.464683413505554, 1), (0.742, 3): (40.18435287475586, 1), (0.742, 1): (49.462045311927795, 1), (0.742, 0.7): (46.082953095436096, 1), (0.742, 0.3): (48.4729288816452, 1), (0.742, 0.1): (58.14574956893921, 1), (0.763, 10): (44.013169169425964, 1), (0.763, 7): (54.11302638053894, 1), (0.763, 3): (54.09723246097565, 1), (0.763, 1): (50.0945200920105, 1), (0.763, 0.7): (35.75268578529358, 1), (0.763, 0.3): (114.38521754741669, 1), (0.763, 0.1): (441.3676174879074, 1), (0.784, 10): (51.18807756900787, 1), (0.784, 7): (51.10437202453613, 1), (0.784, 3): (55.18652856349945, 1), (0.784, 1): (83.77439677715302, 1), (0.784, 0.7): (115.25058233737946, 1), (0.784, 0.3): (183.9370173215866, 1), (0.784, 0.1): (594.2378220558167, 2), (0.805, 10): (48.57236564159393, 1), (0.805, 7): (53.88300621509552, 1), (0.805, 3): (53.669026255607605, 1), (0.805, 1): (88.21507394313812, 1), (0.805, 0.7): (148.42880988121033, 1), (0.805, 0.3): (335.9956910610199, 1), (0.8260000000000001, 10): (72.13280510902405, 1), (0.8260000000000001, 7): (47.92296874523163, 1), (0.8260000000000001, 3): (72.71002912521362, 1), (0.8260000000000001, 1): (180.487734913826, 1), (0.8260000000000001, 0.7): (206.42830348014832, 1), (0.8260000000000001, 0.3): (105.95227551460266, 2), (0.8470000000000001, 10): (51.83043909072876, 1), (0.8470000000000001, 7): (53.340118408203125, 1), (0.8470000000000001, 3): (55.92174315452576, 1), (0.8470000000000001, 1): (494.1976456642151, 2), (0.8470000000000001, 0.7): (345.9749438762665, 2), (0.8680000000000001, 10): (68.41055738925934, 1), (0.8680000000000001, 7): (55.73410618305206, 1), (0.8680000000000001, 3): (118.59046971797943, 2), (0.8680000000000001, 1): (245.23703050613403, 5), (0.8680000000000001, 0.7): (300.21091961860657, 3), (0.8890000000000001, 10): (69.90541923046112, 1), (0.8890000000000001, 7): (77.0376672744751, 1)}

accuracies = []
privacy = []
strategy= []
searchtime = []

map_real_id_to_unique = {}
map_unique_to_name = {}





for k,v in data.items():
	accuracies.append(round(k[0], 2))
	privacy.append(round(k[1], 2))
	searchtime.append(round(v[0], 2))
	strategy.append(v[1])

print(map_unique_to_name)

df = pd.DataFrame({'Minimum Accuracy': accuracies, 'Privacy Epsilon': privacy, 'Search time': searchtime, 'Fastest Strategy': strategy})


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
