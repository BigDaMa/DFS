import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np
from graphviz import Digraph
import tempfile
from fastfeature.plotting.inheritance.tree.MyNode import MyNode
import matplotlib.pyplot as plt
import numpy as np


#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'

# hearts
#file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
#names_all_file = '/home/felix/phd/fastfeature_logs/newest_28_11/name2id.p'
#all_candidates_file = '/home/felix/phd/fastfeature_logs/newest_28_11/all_candidates.p'

# hearts - all features + fi
file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/chart.p'
names_all_file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/name2id.p'
all_candidates_file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/all_candiates.p'


# diabetes
#file = '/home/felix/phd/fastfeatures/results/diabetes/chart.p'
#names_all_file = '/home/felix/phd/fastfeatures/results/diabetes/name2id.p'
#all_candidates_file = '/home/felix/phd/fastfeatures/results/diabetes/all_candiates.p'




all_data = pickle.load(open(file, "rb"))
names_all: Dict[str, int] = pickle.load(open(names_all_file, "rb"))




all_candidates: List[CandidateFeature] = pickle.load(open(all_candidates_file, "rb"))

c = all_candidates[-14]
print(c)
print(all_data['new_scores'][names_all[c.get_name()]])

interpretability_pruned = []
new_scores_pruned = []
names_pruned = []

interpretability_not_pruned = []
new_scores_not_pruned = []
names_not_pruned = []



def depth_greater2(c: CandidateFeature):
    return c.get_transformation_depth() > 2

def do_not_use_feature(c: CandidateFeature):
    return 'resting_electrocardiographic_results' in c.get_name()



print(all_data['names'][0])
print(all_data['ids'][0])
print(names_all['age'])

candidate_id_to_stored_id= {}

for stored_id in range(len(all_data['ids'])):
    candidate_id_to_stored_id[all_data['ids'][stored_id]] = stored_id


for c in all_candidates:
    name = c.get_name()

    if do_not_use_feature(c):
        interpretability_pruned.append(all_data['interpretability'][candidate_id_to_stored_id[names_all[name]]])
        new_scores_pruned.append(all_data['new_scores'][candidate_id_to_stored_id[names_all[name]]])
        names_pruned.append(all_data['names'][candidate_id_to_stored_id[names_all[name]]])
        print(names_pruned[-1] + " : " + str(name))
    else:
        interpretability_not_pruned.append(all_data['interpretability'][candidate_id_to_stored_id[names_all[name]]])
        new_scores_not_pruned.append(all_data['new_scores'][candidate_id_to_stored_id[names_all[name]]])
        names_not_pruned.append(all_data['names'][candidate_id_to_stored_id[names_all[name]]])



def cool_plotting(interpretability_pruned,
              new_scores_pruned,
              names_pruned,
              interpretability_not_pruned,
              new_scores_not_pruned,
              names_not_pruned, start_score, accuracy_lim=None):

    fig, ax = plt.subplots()

    sc_pruned = plt.scatter(interpretability_pruned, new_scores_pruned, c='red')
    sc_not_pruned = plt.scatter(interpretability_not_pruned, new_scores_not_pruned, c='blue')


    ax.set_xlabel("Interpretability (low -> high)")
    ax.set_ylabel("Accuracy (Micro AUC)")
    ax.set_xlim((0, 1.0))
    if type(accuracy_lim) != type(None):
        ax.set_ylim(accuracy_lim)

    ax.axhline(y=start_score, color='red', linestyle='--')

    ax.legend(loc='upper left')

    annot = ax.annotate("", xy=(0,0), xytext=(-220,70),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)


    def update_annot(ind, sc, names):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = names[ind["ind"][0]]
        #print(text + ": " + str(y[ind["ind"][0]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc_pruned.contains(event)
            if cont:
                update_annot(ind, sc_pruned,names_pruned)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                cont, ind = sc_not_pruned.contains(event)
                if cont:
                    update_annot(ind, sc_not_pruned, names_not_pruned)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


    plt.show()

cool_plotting(interpretability_pruned,
              new_scores_pruned,
              names_pruned,
              interpretability_not_pruned,
              new_scores_not_pruned,
              names_not_pruned,
              0.0,
              [0.0, 1.0])








