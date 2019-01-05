import pickle
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes


def cool_plotting(x_good, y_good, names_good, x_bad, y_bad, names_bad, start_score, accuracy_lim=None):

    fig, ax = plt.subplots()
    sc1: Axes.scatter = plt.scatter(x_good, y_good, c='blue')

    sc2: Axes.scatter = plt.scatter(x_bad, y_bad, c='red')

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

    def update_annot(ind, scatter, names):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = names[ind["ind"][0]]
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            scatter = None
            names = None
            cont= None
            ind = None
            print(str(sc1.contains(event)) + " vs " + str(sc2.contains(event)))
            if sc1.contains(event)[0]:
                cont, ind = sc1.contains(event)
                scatter = sc1
                names = names_good
            else:
                cont, ind = sc2.contains(event)
                scatter = sc2
                names = names_bad
            if cont:
                update_annot(ind, scatter, names)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

file = "/home/felix/phd/fastfeature_logs/charts_new/chart_all_no_hyper_opt.p"
names_of_file = "/home/felix/phd/fastfeature_logs/charts_new/name2id_all_no_hyper_op.p"
t_file = '/tmp/ids_to_be_skipped.p'

skipped_all_data = pickle.load(open(t_file, "rb"))
all_data = pickle.load(open(file, "rb"))
names_data = pickle.load(open(names_of_file, "rb"))

interpretability_good = []
score_good = []
names_good = []

interpretability_bad = []
score_bad = []
names_bad = []


to_be_skipped = skipped_all_data['ids_to_be_skipped']
skip_names = skipped_all_data['names']
interpretability_scores = all_data['interpretability']
ids = all_data['ids']

id_2_names = {v: k for k, v in names_data.items()}


for i in range(len(interpretability_scores)):
        assert id_2_names[ids[i]] == all_data['names'][i].split(':')[0]

        if ids[i] not in to_be_skipped:
            interpretability_good.append(all_data['interpretability'][i])
            score_good.append(all_data['new_scores'][i])
            names_good.append(all_data['names'][i] + "_" + id_2_names[ids[i]])
        else:
            assert skip_names[ids[i]] == id_2_names[ids[i]]
            interpretability_bad.append(all_data['interpretability'][i])
            score_bad.append(all_data['new_scores'][i])
            names_bad.append(all_data['names'][i])



print(names_bad)


cool_plotting(interpretability_good,
              score_good,
              names_good,
              interpretability_bad,
              score_bad,
              names_bad,
              all_data['start_score'])