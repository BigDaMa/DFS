import matplotlib.pyplot as plt
import numpy as np


def calculate_cummulative(x,y):
    ids = np.argsort(np.array(x)*-1.0)
    sorted_scores = np.array(y)[ids] # 1.0 -> 0.0

    cummulative = []
    cur_max = -10
    for i in range(len(sorted_scores)):
        if cur_max < sorted_scores[i]:
            cur_max = sorted_scores[i]
        cummulative.append(cur_max)

    return np.array(x)[ids], cummulative




def cool_plotting(x, y, names, start_score, accuracy_lim=None):

    fig, ax = plt.subplots()


    cum_x, cum_y = calculate_cummulative(x,y)
    plt.scatter(cum_x, cum_y)

    sc = plt.scatter(x, y)


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

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = names[ind["ind"][0]]
        print(text + ": " + str(y[ind["ind"][0]]))
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

    plt.show()