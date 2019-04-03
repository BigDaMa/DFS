import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from typing import List, Dict, Set
import copy
from fastsklearnfeature.transformations.Transformation import Transformation
from fastsklearnfeature.transformations.UnaryTransformation import UnaryTransformation
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.candidates.RawFeature import RawFeature
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

#path = '/home/felix/phd/fastfeatures/results/11_03_incremental_construction'
#path = '/home/felix/phd/fastfeatures/results/12_03_incremental_03_threshold'
#path = '/home/felix/phd/fastfeatures/results/12_03_incremental_02_threshold'
#path = '/tmp'
#path = '/home/felix/phd/fastfeatures/results/15_03_timed_transfusion'
#path = '/home/felix/phd/fastfeatures/results/15_03_timed_transfusion_node1'
#path = '/home/felix/phd/fastfeatures/results/16_03_test_transfusion_me'
#path = '/home/felix/phd/fastfeatures/results/18_03_banknote'
#path = '/home/felix/phd/fastfeatures/results/18_03_iris'
#path = '/home/felix/phd/fastfeatures/results/20_03_transfusion'
path = '/home/felix/phd/fastfeatures/results/2_4_transfusion_with_predictions'


cost_2_raw_features: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_raw.p", "rb"))
cost_2_unary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_unary.p", "rb"))
cost_2_binary_transformed: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_binary.p", "rb"))
cost_2_combination: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_combination.p", "rb"))
cost_2_dropped_evaluated_candidates: Dict[int, List[CandidateFeature]] = pickle.load(open(path + "/data_dropped.p", "rb"))



def crawl_predictions(data: Dict[int, List[CandidateFeature]], names, complexity, predictions):
    for key, my_list in data.items():
        for c in my_list:
            if c.runtime_properties['score'] >= 0.0 and 'predictions' in c.runtime_properties:
                names.append(str(c))
                complexity.append(c.get_complexity())
                predictions.append(c.runtime_properties['predictions'])


names = []
complexity = []
predictions = []

crawl_predictions(cost_2_raw_features, names, complexity, predictions)
crawl_predictions(cost_2_unary_transformed, names, complexity, predictions)
crawl_predictions(cost_2_binary_transformed, names, complexity, predictions)
crawl_predictions(cost_2_combination, names, complexity, predictions)
crawl_predictions(cost_2_dropped_evaluated_candidates, names, complexity, predictions)


true_pediction = np.load(path + "/true_predictions.npy")
predictions.append(true_pediction)
names.append('True Predictions')

X = np.matrix(predictions)


tsne = TSNE(n_components=2, init='random', #method='barnes_hut',
                         random_state=0, learning_rate=1000, n_iter=1000,
                         verbose=2)
Y = tsne.fit_transform(X)

fig, (ax) = plt.subplots(1, 1)

plts = []
labels = []

sc = ax.scatter(Y[:, 0], Y[:, 1], picker=5)

for n_i in range(len(names)):
    if names[n_i] == 'Frequency' or \
       names[n_i] == 'true_divide(Recency,Monetary)' or \
       names[n_i] == 'true_divide((nanmin(Time) GroupyBy Recency),Monetary)':
        ax.scatter([Y[n_i, 0]], [Y[n_i, 1]], picker=5, color='red')
    if names[n_i] == 'True Predictions':
        ax.scatter([Y[n_i, 0]], [Y[n_i, 1]], picker=5, color='green')



'''
ax.legend(plts, labels,
          scatterpoints=1,
          loc='lower left',
          ncol=3,
          fontsize=8)
'''

norm = plt.Normalize(1,4)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" | ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.set_window_title('t-SNE Browser')


plt.show()



