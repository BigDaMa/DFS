import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

accuracy_ranking = pickle.load(open("/home/felix/phd/ranking_exeriments/accuracy_ranking.p", "rb"))


fairness_ranking = pickle.load(open("/home/felix/phd/ranking_exeriments/fairness_ranking.p", "rb"))
fairness_ranking *= -1.0


robustness_ranking = pickle.load(open("/home/felix/phd/ranking_exeriments/robustness_ranking.p", "rb"))
robustness_ranking *= -1.0


names = pickle.load(open("/home/felix/phd/ranking_exeriments/names.p", "rb"))

from scipy.stats import spearmanr

print(spearmanr(fairness_ranking, accuracy_ranking))
print(spearmanr(fairness_ranking, robustness_ranking))
print(spearmanr(accuracy_ranking, robustness_ranking))