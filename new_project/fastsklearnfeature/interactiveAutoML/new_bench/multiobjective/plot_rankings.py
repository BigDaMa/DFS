import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

accuracy_ranking = pickle.load(open("/home/felix/phd/ranking_exeriments/accuracy_ranking.p", "rb"))
print(accuracy_ranking)
accuracy_ranking /= np.sum(accuracy_ranking)
#accuracy_ranking = scaler.fit_transform(accuracy_ranking.reshape(-1, 1))

fairness_ranking = pickle.load(open("/home/felix/phd/ranking_exeriments/fairness_ranking.p", "rb"))
print(fairness_ranking)
fairness_ranking /= np.sum(fairness_ranking)
#fairness_ranking = scaler.fit_transform(fairness_ranking.reshape(-1, 1))

robustness_ranking = pickle.load(open("/home/felix/phd/ranking_exeriments/robustness_ranking.p", "rb"))
print(robustness_ranking)
robustness_ranking /= np.sum(robustness_ranking)
#robustness_ranking = scaler.fit_transform(robustness_ranking.reshape(-1, 1))


names = pickle.load(open("/home/felix/phd/ranking_exeriments/names.p", "rb"))



import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ids = np.argsort(accuracy_ranking,axis=0).flatten()
#ids = np.argsort(fairness_ranking,axis=0).flatten()
#ids = np.argsort(robustness_ranking,axis=0).flatten()
print(ids)
labels = np.array(names)[ids]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars



fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracy_ranking.flatten()[ids], width, label='accuracy')
rects2 = ax.bar(x + 0, fairness_ranking.flatten()[ids], width, label='fairness')
rects3 = ax.bar(x + width/2, robustness_ranking.flatten()[ids], width, label='robustness')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()