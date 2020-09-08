import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

from fastsklearnfeature.interactiveAutoML.new_bench.multiobjective.metalearning.analyse.heatmap.heatmap_util import pivot2latex
import pickle
import copy


data = {('Oracle', 'telco-customer-churn'): 0.09090909090909091, ('TPE(Variance)', 'telco-customer-churn'): 0.03409090909090909, ('TPE($\\chi^2$)', 'telco-customer-churn'): 0.022727272727272728, ('TPE(FCBF)', 'telco-customer-churn'): 0.03409090909090909, ('TPE(Fisher)', 'telco-customer-churn'): 0.03409090909090909, ('TPE(MIM)', 'telco-customer-churn'): 0.045454545454545456, ('TPE(MCFS)', 'telco-customer-churn'): 0.03409090909090909, ('TPE(ReliefF)', 'telco-customer-churn'): 0.022727272727272728, ('TPE(NR)', 'telco-customer-churn'): 0.03409090909090909, ('SA(NR)', 'telco-customer-churn'): 0.03409090909090909, ('NSGA-II(NR)', 'telco-customer-churn'): 0.022727272727272728, ('ES(NR)', 'telco-customer-churn'): 0.045454545454545456, ('SFS(NR)', 'telco-customer-churn'): 0.03409090909090909, ('SBS(NR)', 'telco-customer-churn'): 0.0, ('SFFS(NR)', 'telco-customer-churn'): 0.06818181818181818, ('SBFS(NR)', 'telco-customer-churn'): 0.0, ('RFE(Model)', 'telco-customer-churn'): 0.011363636363636364, ('Complete Set', 'telco-customer-churn'): 0.0, ('Metalearning', 'telco-customer-churn'): 0.045454545454545456, ('Oracle', 'socmob'): 0.3793103448275862, ('TPE(Variance)', 'socmob'): 0.26436781609195403, ('TPE($\\chi^2$)', 'socmob'): 0.27586206896551724, ('TPE(FCBF)', 'socmob'): 0.27586206896551724, ('TPE(Fisher)', 'socmob'): 0.20689655172413793, ('TPE(MIM)', 'socmob'): 0.22988505747126436, ('TPE(MCFS)', 'socmob'): 0.2988505747126437, ('TPE(ReliefF)', 'socmob'): 0.26436781609195403, ('TPE(NR)', 'socmob'): 0.27586206896551724, ('SA(NR)', 'socmob'): 0.28735632183908044, ('NSGA-II(NR)', 'socmob'): 0.28735632183908044, ('ES(NR)', 'socmob'): 0.27586206896551724, ('SFS(NR)', 'socmob'): 0.26436781609195403, ('SBS(NR)', 'socmob'): 0.26436781609195403, ('SFFS(NR)', 'socmob'): 0.26436781609195403, ('SBFS(NR)', 'socmob'): 0.21839080459770116, ('RFE(Model)', 'socmob'): 0.14942528735632185, ('Complete Set', 'socmob'): 0.06896551724137931, ('Metalearning', 'socmob'): 0.28735632183908044, ('Oracle', 'ilpd'): 0.07058823529411765, ('TPE(Variance)', 'ilpd'): 0.011764705882352941, ('TPE($\\chi^2$)', 'ilpd'): 0.011764705882352941, ('TPE(FCBF)', 'ilpd'): 0.011764705882352941, ('TPE(Fisher)', 'ilpd'): 0.011764705882352941, ('TPE(MIM)', 'ilpd'): 0.0, ('TPE(MCFS)', 'ilpd'): 0.0, ('TPE(ReliefF)', 'ilpd'): 0.011764705882352941, ('TPE(NR)', 'ilpd'): 0.011764705882352941, ('SA(NR)', 'ilpd'): 0.0, ('NSGA-II(NR)', 'ilpd'): 0.023529411764705882, ('ES(NR)', 'ilpd'): 0.0, ('SFS(NR)', 'ilpd'): 0.023529411764705882, ('SBS(NR)', 'ilpd'): 0.0, ('SFFS(NR)', 'ilpd'): 0.023529411764705882, ('SBFS(NR)', 'ilpd'): 0.023529411764705882, ('RFE(Model)', 'ilpd'): 0.0, ('Complete Set', 'ilpd'): 0.0, ('Metalearning', 'ilpd'): 0.03529411764705882, ('Oracle', 'kdd_internet_usage'): 0.20481927710843373, ('TPE(Variance)', 'kdd_internet_usage'): 0.13253012048192772, ('TPE($\\chi^2$)', 'kdd_internet_usage'): 0.18072289156626506, ('TPE(FCBF)', 'kdd_internet_usage'): 0.1686746987951807, ('TPE(Fisher)', 'kdd_internet_usage'): 0.1927710843373494, ('TPE(MIM)', 'kdd_internet_usage'): 0.1566265060240964, ('TPE(MCFS)', 'kdd_internet_usage'): 0.12048192771084337, ('TPE(ReliefF)', 'kdd_internet_usage'): 0.14457831325301204, ('TPE(NR)', 'kdd_internet_usage'): 0.0963855421686747, ('SA(NR)', 'kdd_internet_usage'): 0.0963855421686747, ('NSGA-II(NR)', 'kdd_internet_usage'): 0.0963855421686747, ('ES(NR)', 'kdd_internet_usage'): 0.012048192771084338, ('SFS(NR)', 'kdd_internet_usage'): 0.012048192771084338, ('SBS(NR)', 'kdd_internet_usage'): 0.03614457831325301, ('SFFS(NR)', 'kdd_internet_usage'): 0.012048192771084338, ('SBFS(NR)', 'kdd_internet_usage'): 0.024096385542168676, ('RFE(Model)', 'kdd_internet_usage'): 0.024096385542168676, ('Complete Set', 'kdd_internet_usage'): 0.024096385542168676, ('Metalearning', 'kdd_internet_usage'): 0.07228915662650602, ('Oracle', 'compas-two-years'): 0.2073170731707317, ('TPE(Variance)', 'compas-two-years'): 0.012195121951219513, ('TPE($\\chi^2$)', 'compas-two-years'): 0.06097560975609756, ('TPE(FCBF)', 'compas-two-years'): 0.07317073170731707, ('TPE(Fisher)', 'compas-two-years'): 0.07317073170731707, ('TPE(MIM)', 'compas-two-years'): 0.08536585365853659, ('TPE(MCFS)', 'compas-two-years'): 0.0975609756097561, ('TPE(ReliefF)', 'compas-two-years'): 0.06097560975609756, ('TPE(NR)', 'compas-two-years'): 0.04878048780487805, ('SA(NR)', 'compas-two-years'): 0.07317073170731707, ('NSGA-II(NR)', 'compas-two-years'): 0.0975609756097561, ('ES(NR)', 'compas-two-years'): 0.14634146341463414, ('SFS(NR)', 'compas-two-years'): 0.17073170731707318, ('SBS(NR)', 'compas-two-years'): 0.06097560975609756, ('SFFS(NR)', 'compas-two-years'): 0.14634146341463414, ('SBFS(NR)', 'compas-two-years'): 0.04878048780487805, ('RFE(Model)', 'compas-two-years'): 0.06097560975609756, ('Complete Set', 'compas-two-years'): 0.012195121951219513, ('Metalearning', 'compas-two-years'): 0.0975609756097561, ('Oracle', 'braziltourism'): 0.6419753086419753, ('TPE(Variance)', 'braziltourism'): 0.2716049382716049, ('TPE($\\chi^2$)', 'braziltourism'): 0.32098765432098764, ('TPE(FCBF)', 'braziltourism'): 0.3333333333333333, ('TPE(Fisher)', 'braziltourism'): 0.25925925925925924, ('TPE(MIM)', 'braziltourism'): 0.37037037037037035, ('TPE(MCFS)', 'braziltourism'): 0.2716049382716049, ('TPE(ReliefF)', 'braziltourism'): 0.24691358024691357, ('TPE(NR)', 'braziltourism'): 0.32098765432098764, ('SA(NR)', 'braziltourism'): 0.24691358024691357, ('NSGA-II(NR)', 'braziltourism'): 0.32098765432098764, ('ES(NR)', 'braziltourism'): 0.5679012345679012, ('SFS(NR)', 'braziltourism'): 0.5802469135802469, ('SBS(NR)', 'braziltourism'): 0.2962962962962963, ('SFFS(NR)', 'braziltourism'): 0.5679012345679012, ('SBFS(NR)', 'braziltourism'): 0.2222222222222222, ('RFE(Model)', 'braziltourism'): 0.19753086419753085, ('Complete Set', 'braziltourism'): 0.1111111111111111, ('Metalearning', 'braziltourism'): 0.5061728395061729, ('Oracle', 'credit-g'): 0.275, ('TPE(Variance)', 'credit-g'): 0.15, ('TPE($\\chi^2$)', 'credit-g'): 0.175, ('TPE(FCBF)', 'credit-g'): 0.15, ('TPE(Fisher)', 'credit-g'): 0.175, ('TPE(MIM)', 'credit-g'): 0.15, ('TPE(MCFS)', 'credit-g'): 0.1375, ('TPE(ReliefF)', 'credit-g'): 0.1875, ('TPE(NR)', 'credit-g'): 0.1375, ('SA(NR)', 'credit-g'): 0.125, ('NSGA-II(NR)', 'credit-g'): 0.1625, ('ES(NR)', 'credit-g'): 0.225, ('SFS(NR)', 'credit-g'): 0.2375, ('SBS(NR)', 'credit-g'): 0.075, ('SFFS(NR)', 'credit-g'): 0.2375, ('SBFS(NR)', 'credit-g'): 0.075, ('RFE(Model)', 'credit-g'): 0.1125, ('Complete Set', 'credit-g'): 0.0375, ('Metalearning', 'credit-g'): 0.2125, ('Oracle', '1m'): 0.35, ('TPE(Variance)', '1m'): 0.1875, ('TPE($\\chi^2$)', '1m'): 0.1875, ('TPE(FCBF)', '1m'): 0.175, ('TPE(Fisher)', '1m'): 0.1875, ('TPE(MIM)', '1m'): 0.225, ('TPE(MCFS)', '1m'): 0.15, ('TPE(ReliefF)', '1m'): 0.1875, ('TPE(NR)', '1m'): 0.2375, ('SA(NR)', '1m'): 0.25, ('NSGA-II(NR)', '1m'): 0.2125, ('ES(NR)', '1m'): 0.1875, ('SFS(NR)', '1m'): 0.2125, ('SBS(NR)', '1m'): 0.1125, ('SFFS(NR)', '1m'): 0.2375, ('SBFS(NR)', '1m'): 0.1125, ('RFE(Model)', '1m'): 0.175, ('Complete Set', '1m'): 0.1, ('Metalearning', '1m'): 0.2375, ('Oracle', 'adult'): 0.08974358974358974, ('TPE(Variance)', 'adult'): 0.0641025641025641, ('TPE($\\chi^2$)', 'adult'): 0.07692307692307693, ('TPE(FCBF)', 'adult'): 0.0641025641025641, ('TPE(Fisher)', 'adult'): 0.07692307692307693, ('TPE(MIM)', 'adult'): 0.08974358974358974, ('TPE(MCFS)', 'adult'): 0.01282051282051282, ('TPE(ReliefF)', 'adult'): 0.038461538461538464, ('TPE(NR)', 'adult'): 0.05128205128205128, ('SA(NR)', 'adult'): 0.038461538461538464, ('NSGA-II(NR)', 'adult'): 0.0641025641025641, ('ES(NR)', 'adult'): 0.01282051282051282, ('SFS(NR)', 'adult'): 0.01282051282051282, ('SBS(NR)', 'adult'): 0.01282051282051282, ('SFFS(NR)', 'adult'): 0.02564102564102564, ('SBFS(NR)', 'adult'): 0.01282051282051282, ('RFE(Model)', 'adult'): 0.0, ('Complete Set', 'adult'): 0.0, ('Metalearning', 'adult'): 0.0641025641025641, ('Oracle', 'sick'): 0.2631578947368421, ('TPE(Variance)', 'sick'): 0.05263157894736842, ('TPE($\\chi^2$)', 'sick'): 0.09210526315789473, ('TPE(FCBF)', 'sick'): 0.18421052631578946, ('TPE(Fisher)', 'sick'): 0.18421052631578946, ('TPE(MIM)', 'sick'): 0.14473684210526316, ('TPE(MCFS)', 'sick'): 0.06578947368421052, ('TPE(ReliefF)', 'sick'): 0.21052631578947367, ('TPE(NR)', 'sick'): 0.09210526315789473, ('SA(NR)', 'sick'): 0.10526315789473684, ('NSGA-II(NR)', 'sick'): 0.09210526315789473, ('ES(NR)', 'sick'): 0.07894736842105263, ('SFS(NR)', 'sick'): 0.15789473684210525, ('SBS(NR)', 'sick'): 0.07894736842105263, ('SFFS(NR)', 'sick'): 0.15789473684210525, ('SBFS(NR)', 'sick'): 0.06578947368421052, ('RFE(Model)', 'sick'): 0.10526315789473684, ('Complete Set', 'sick'): 0.02631578947368421, ('Metalearning', 'sick'): 0.17105263157894737, ('Oracle', 'ipums_la_99-small'): 0.92, ('TPE(Variance)', 'ipums_la_99-small'): 0.6, ('TPE($\\chi^2$)', 'ipums_la_99-small'): 0.56, ('TPE(FCBF)', 'ipums_la_99-small'): 0.5733333333333334, ('TPE(Fisher)', 'ipums_la_99-small'): 0.5333333333333333, ('TPE(MIM)', 'ipums_la_99-small'): 0.5466666666666666, ('TPE(MCFS)', 'ipums_la_99-small'): 0.5866666666666667, ('TPE(ReliefF)', 'ipums_la_99-small'): 0.6133333333333333, ('TPE(NR)', 'ipums_la_99-small'): 0.56, ('SA(NR)', 'ipums_la_99-small'): 0.56, ('NSGA-II(NR)', 'ipums_la_99-small'): 0.5866666666666667, ('ES(NR)', 'ipums_la_99-small'): 0.88, ('SFS(NR)', 'ipums_la_99-small'): 0.88, ('SBS(NR)', 'ipums_la_99-small'): 0.28, ('SFFS(NR)', 'ipums_la_99-small'): 0.88, ('SBFS(NR)', 'ipums_la_99-small'): 0.24, ('RFE(Model)', 'ipums_la_99-small'): 0.37333333333333335, ('Complete Set', 'ipums_la_99-small'): 0.24, ('Metalearning', 'ipums_la_99-small'): 0.7333333333333333, ('Oracle', 'DiabeticMellitus'): 0.48, ('TPE(Variance)', 'DiabeticMellitus'): 0.14666666666666667, ('TPE($\\chi^2$)', 'DiabeticMellitus'): 0.14666666666666667, ('TPE(FCBF)', 'DiabeticMellitus'): 0.16, ('TPE(Fisher)', 'DiabeticMellitus'): 0.17333333333333334, ('TPE(MIM)', 'DiabeticMellitus'): 0.18666666666666668, ('TPE(MCFS)', 'DiabeticMellitus'): 0.14666666666666667, ('TPE(ReliefF)', 'DiabeticMellitus'): 0.21333333333333335, ('TPE(NR)', 'DiabeticMellitus'): 0.16, ('SA(NR)', 'DiabeticMellitus'): 0.12, ('NSGA-II(NR)', 'DiabeticMellitus'): 0.21333333333333335, ('ES(NR)', 'DiabeticMellitus'): 0.3333333333333333, ('SFS(NR)', 'DiabeticMellitus'): 0.36, ('SBS(NR)', 'DiabeticMellitus'): 0.09333333333333334, ('SFFS(NR)', 'DiabeticMellitus'): 0.36, ('SBFS(NR)', 'DiabeticMellitus'): 0.08, ('RFE(Model)', 'DiabeticMellitus'): 0.13333333333333333, ('Complete Set', 'DiabeticMellitus'): 0.02666666666666667, ('Metalearning', 'DiabeticMellitus'): 0.32, ('Oracle', 'SpeedDating'): 0.0, ('TPE(Variance)', 'SpeedDating'): 0.0, ('TPE($\\chi^2$)', 'SpeedDating'): 0.0, ('TPE(FCBF)', 'SpeedDating'): 0.0, ('TPE(Fisher)', 'SpeedDating'): 0.0, ('TPE(MIM)', 'SpeedDating'): 0.0, ('TPE(MCFS)', 'SpeedDating'): 0.0, ('TPE(ReliefF)', 'SpeedDating'): 0.0, ('TPE(NR)', 'SpeedDating'): 0.0, ('SA(NR)', 'SpeedDating'): 0.0, ('NSGA-II(NR)', 'SpeedDating'): 0.0, ('ES(NR)', 'SpeedDating'): 0.0, ('SFS(NR)', 'SpeedDating'): 0.0, ('SBS(NR)', 'SpeedDating'): 0.0, ('SFFS(NR)', 'SpeedDating'): 0.0, ('SBFS(NR)', 'SpeedDating'): 0.0, ('RFE(Model)', 'SpeedDating'): 0.0, ('Complete Set', 'SpeedDating'): 0.0, ('Metalearning', 'SpeedDating'): 0.0, ('Oracle', 'primary-tumor'): 0.136986301369863, ('TPE(Variance)', 'primary-tumor'): 0.0410958904109589, ('TPE($\\chi^2$)', 'primary-tumor'): 0.0684931506849315, ('TPE(FCBF)', 'primary-tumor'): 0.0684931506849315, ('TPE(Fisher)', 'primary-tumor'): 0.0821917808219178, ('TPE(MIM)', 'primary-tumor'): 0.0547945205479452, ('TPE(MCFS)', 'primary-tumor'): 0.0273972602739726, ('TPE(ReliefF)', 'primary-tumor'): 0.0958904109589041, ('TPE(NR)', 'primary-tumor'): 0.0547945205479452, ('SA(NR)', 'primary-tumor'): 0.0821917808219178, ('NSGA-II(NR)', 'primary-tumor'): 0.0684931506849315, ('ES(NR)', 'primary-tumor'): 0.0547945205479452, ('SFS(NR)', 'primary-tumor'): 0.0547945205479452, ('SBS(NR)', 'primary-tumor'): 0.0273972602739726, ('SFFS(NR)', 'primary-tumor'): 0.0547945205479452, ('SBFS(NR)', 'primary-tumor'): 0.0410958904109589, ('RFE(Model)', 'primary-tumor'): 0.0410958904109589, ('Complete Set', 'primary-tumor'): 0.0136986301369863, ('Metalearning', 'primary-tumor'): 0.0547945205479452, ('Oracle', 'pbcseq'): 0.3150684931506849, ('TPE(Variance)', 'pbcseq'): 0.136986301369863, ('TPE($\\chi^2$)', 'pbcseq'): 0.1643835616438356, ('TPE(FCBF)', 'pbcseq'): 0.1232876712328767, ('TPE(Fisher)', 'pbcseq'): 0.1506849315068493, ('TPE(MIM)', 'pbcseq'): 0.1095890410958904, ('TPE(MCFS)', 'pbcseq'): 0.1232876712328767, ('TPE(ReliefF)', 'pbcseq'): 0.0821917808219178, ('TPE(NR)', 'pbcseq'): 0.136986301369863, ('SA(NR)', 'pbcseq'): 0.1095890410958904, ('NSGA-II(NR)', 'pbcseq'): 0.1643835616438356, ('ES(NR)', 'pbcseq'): 0.1780821917808219, ('SFS(NR)', 'pbcseq'): 0.1917808219178082, ('SBS(NR)', 'pbcseq'): 0.0684931506849315, ('SFFS(NR)', 'pbcseq'): 0.1780821917808219, ('SBFS(NR)', 'pbcseq'): 0.0547945205479452, ('RFE(Model)', 'pbcseq'): 0.0547945205479452, ('Complete Set', 'pbcseq'): 0.0547945205479452, ('Metalearning', 'pbcseq'): 0.1917808219178082, ('Oracle', 'arrhythmia'): 0.24285714285714285, ('TPE(Variance)', 'arrhythmia'): 0.08571428571428572, ('TPE($\\chi^2$)', 'arrhythmia'): 0.08571428571428572, ('TPE(FCBF)', 'arrhythmia'): 0.1, ('TPE(Fisher)', 'arrhythmia'): 0.1, ('TPE(MIM)', 'arrhythmia'): 0.11428571428571428, ('TPE(MCFS)', 'arrhythmia'): 0.1, ('TPE(ReliefF)', 'arrhythmia'): 0.14285714285714285, ('TPE(NR)', 'arrhythmia'): 0.1, ('SA(NR)', 'arrhythmia'): 0.15714285714285714, ('NSGA-II(NR)', 'arrhythmia'): 0.08571428571428572, ('ES(NR)', 'arrhythmia'): 0.14285714285714285, ('SFS(NR)', 'arrhythmia'): 0.15714285714285714, ('SBS(NR)', 'arrhythmia'): 0.08571428571428572, ('SFFS(NR)', 'arrhythmia'): 0.15714285714285714, ('SBFS(NR)', 'arrhythmia'): 0.08571428571428572, ('RFE(Model)', 'arrhythmia'): 0.11428571428571428, ('Complete Set', 'arrhythmia'): 0.04285714285714286, ('Metalearning', 'arrhythmia'): 0.15714285714285714, ('Oracle', 'irish'): 0.38235294117647056, ('TPE(Variance)', 'irish'): 0.23529411764705882, ('TPE($\\chi^2$)', 'irish'): 0.2647058823529412, ('TPE(FCBF)', 'irish'): 0.2647058823529412, ('TPE(Fisher)', 'irish'): 0.2647058823529412, ('TPE(MIM)', 'irish'): 0.25, ('TPE(MCFS)', 'irish'): 0.11764705882352941, ('TPE(ReliefF)', 'irish'): 0.2647058823529412, ('TPE(NR)', 'irish'): 0.29411764705882354, ('SA(NR)', 'irish'): 0.29411764705882354, ('NSGA-II(NR)', 'irish'): 0.27941176470588236, ('ES(NR)', 'irish'): 0.20588235294117646, ('SFS(NR)', 'irish'): 0.23529411764705882, ('SBS(NR)', 'irish'): 0.23529411764705882, ('SFFS(NR)', 'irish'): 0.25, ('SBFS(NR)', 'irish'): 0.25, ('RFE(Model)', 'irish'): 0.23529411764705882, ('Complete Set', 'irish'): 0.08823529411764706, ('Metalearning', 'irish'): 0.25, ('Oracle', 'Titanic'): 0.24096385542168675, ('TPE(Variance)', 'Titanic'): 0.1686746987951807, ('TPE($\\chi^2$)', 'Titanic'): 0.14457831325301204, ('TPE(FCBF)', 'Titanic'): 0.10843373493975904, ('TPE(Fisher)', 'Titanic'): 0.10843373493975904, ('TPE(MIM)', 'Titanic'): 0.10843373493975904, ('TPE(MCFS)', 'Titanic'): 0.03614457831325301, ('TPE(ReliefF)', 'Titanic'): 0.10843373493975904, ('TPE(NR)', 'Titanic'): 0.14457831325301204, ('SA(NR)', 'Titanic'): 0.12048192771084337, ('NSGA-II(NR)', 'Titanic'): 0.12048192771084337, ('ES(NR)', 'Titanic'): 0.1686746987951807, ('SFS(NR)', 'Titanic'): 0.1686746987951807, ('SBS(NR)', 'Titanic'): 0.03614457831325301, ('SFFS(NR)', 'Titanic'): 0.1686746987951807, ('SBFS(NR)', 'Titanic'): 0.03614457831325301, ('RFE(Model)', 'Titanic'): 0.07228915662650602, ('Complete Set', 'Titanic'): 0.03614457831325301, ('Metalearning', 'Titanic'): 0.18072289156626506, ('Oracle', 'AirlinesCodrnaAdult'): 0.09090909090909091, ('TPE(Variance)', 'AirlinesCodrnaAdult'): 0.07954545454545454, ('TPE($\\chi^2$)', 'AirlinesCodrnaAdult'): 0.056818181818181816, ('TPE(FCBF)', 'AirlinesCodrnaAdult'): 0.045454545454545456, ('TPE(Fisher)', 'AirlinesCodrnaAdult'): 0.0, ('TPE(MIM)', 'AirlinesCodrnaAdult'): 0.022727272727272728, ('TPE(MCFS)', 'AirlinesCodrnaAdult'): 0.0, ('TPE(ReliefF)', 'AirlinesCodrnaAdult'): 0.0, ('TPE(NR)', 'AirlinesCodrnaAdult'): 0.06818181818181818, ('SA(NR)', 'AirlinesCodrnaAdult'): 0.06818181818181818, ('NSGA-II(NR)', 'AirlinesCodrnaAdult'): 0.07954545454545454, ('ES(NR)', 'AirlinesCodrnaAdult'): 0.06818181818181818, ('SFS(NR)', 'AirlinesCodrnaAdult'): 0.06818181818181818, ('SBS(NR)', 'AirlinesCodrnaAdult'): 0.022727272727272728, ('SFFS(NR)', 'AirlinesCodrnaAdult'): 0.06818181818181818, ('SBFS(NR)', 'AirlinesCodrnaAdult'): 0.022727272727272728, ('RFE(Model)', 'AirlinesCodrnaAdult'): 0.022727272727272728, ('Complete Set', 'AirlinesCodrnaAdult'): 0.03409090909090909, ('Metalearning', 'AirlinesCodrnaAdult'): 0.06818181818181818}


strategies = []
datasets = []
coverage = []

map_real_id_to_unique = {}
map_unique_to_name = {}




for k,v in data.items():
	if k[1] != 'SpeedDating':	
		strategies.append(k[0])
		datasets.append(k[1])
		if v > 0:
			coverage.append(float(v) / data[('Oracle', k[1])])
		else:
			coverage.append(float(v))

print(map_unique_to_name)

df = pd.DataFrame({'Strategies': strategies,
				   'Datasets': datasets,
				   'Coverage': coverage})


my_pivot = df.pivot("Strategies", "Datasets", "Coverage")


'''
1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher)',
			 5: 'TPE(MIM)',
			 6: 'TPE(MCFS)',
			 7: 'TPE(ReliefF)',
			 8: 'TPE(NR)',
             9: 'SA(NR)',
			 10: 'NSGA-II(NR)',
			 
'''



my_pivot = my_pivot.reindex(['Complete Set',
								  'ES(NR)',
								  'SFS(NR)',
								  'SBS(NR)',
								  'SFFS(NR)',
								  'SBFS(NR)',
								  'RFE(Model)',
								  'TPE(Fisher)',
								  'TPE(ReliefF)',
								  'TPE(MIM)',
								  'TPE(FCBF)',
								  'TPE(MCFS)',
								  'TPE(Variance)',
								  'TPE($\chi^2$)',
								  'TPE(NR)',
								  'SA(NR)',
								  'NSGA-II(NR)',
								  'Metalearning',
								  'Oracle'
								  ])

my_pivot = my_pivot.reindex(['Complete Set',
								  'SBFS(NR)',
                                  'SBS(NR)',
								  'RFE(Model)',
								  'TPE(MCFS)',
								  'TPE(Variance)',
							      'TPE(ReliefF)',
								  'SA(NR)',
                                  'TPE(NR)',
								  'TPE(Fisher)',
                                  'TPE(MIM)',
							      'TPE(FCBF)',
								  'TPE($\chi^2$)',
								  'NSGA-II(NR)',
								  'ES(NR)',
								  'SFS(NR)',
								  'SFFS(NR)',
								  'Metalearning',
								  'Oracle'
								  ])

print(my_pivot)

real_labels = []
for strategy_i in range(len(my_pivot.axes[0])):
	row_labels = []
	for data_i in range(len(my_pivot.axes[1])):
		row_labels.append(round(data[(my_pivot.axes[0][strategy_i], my_pivot.axes[1][data_i])],2))
	real_labels.append(row_labels)


sns.set_context("paper", rc={"font.size":5, "axes.labelsize":5, 'text.usetex': True})
ax = sns.heatmap(my_pivot, annot=real_labels, annot_kws={"fontsize":5})
plt.tight_layout()
plt.savefig("output.pdf", dpi=300)
#plt.show()



'''

fig, ax = plt.subplots(figsize=(3.5, 3.5))

print(type(my_pivot))

print(my_pivot.axes)






sns.heatmap(my_pivot, cbar=False, ax=ax)


pivot2latex(my_pivot)
'''