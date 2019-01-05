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

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
file = '/home/felix/phd/fastfeature_logs/charts/chart_all_sorted_by_complexity_fold20_hyper_opt_1045min.p'

t_file = '/tmp/ids_to_be_skipped.p'


missed_data = pickle.load(open('/home/felix/phd/fastfeature_logs/charts/missed_ids.p', "rb"))
skipped_all_data = pickle.load(open(t_file, "rb"))
all_data = pickle.load(open(file, "rb"))

names_data = pickle.load(open('/tmp/names.p', "rb"))


interpretability_good = []
score_good = []
names_good = []

interpretability_bad = []
score_bad = []
names_bad = []

counter = 0
for i in range(35594):
    if not i in missed_data['missed_ids']:
        print('#' + str(all_data['names'][counter].split(':')[0]) + "# vs #" + str(names_data['names'][i]) +'#')
        assert all_data['names'][counter].split(':')[0] == names_data['names'][i], "sorting is not deterministic"


        if i not in skipped_all_data['ids_to_be_skipped']:
            interpretability_good.append(all_data['interpretability'][counter])
            score_good.append(all_data['new_scores'][counter])
            names_good.append(all_data['names'][counter])
        else:
            interpretability_bad.append(all_data['interpretability'][counter])
            score_bad.append(all_data['new_scores'][counter])
            names_bad.append(all_data['names'][counter])


        counter +=1


print(names_bad)


cool_plotting(interpretability_good,
              score_good,
              names_good,
              interpretability_bad,
              score_bad,
              names_bad,
              all_data['start_score'])