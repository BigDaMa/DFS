import pickle
from fastfeature.plotting.plotter import cool_plotting

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_sorted_by_complexity_fold20_hyper_opt_1045min.p'

#heart
#file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
#my_range = (0.72, 0.88)
# heart also raw features
file = '/home/felix/phd/fastfeatures/results/heart_also_raw_features/chart.p'
my_range = (0.50, 0.88)



#diabetes
#file = '/home/felix/phd/fastfeatures/results/diabetes/chart.p'
#my_range = (0.72, 0.78)

all_data = pickle.load(open(file, "rb"))

interpretability = all_data['interpretability']
scores = all_data['new_scores']

#calculate cover

#get first local optima



print(interpretability)

import numpy as np
from scipy.signal import argrelextrema


ids = np.argsort(np.array(interpretability) * -1.0)
print("sorted " + str(interpretability[ids[0]]))

sorted_scores = np.array(scores)[ids]

cummulative = []
cur_max =-10
for i in range(len(sorted_scores)):
    if cur_max < sorted_scores[i]:
        cur_max = sorted_scores[i]
    cummulative.append(cur_max)

cum_flip = cummulative #np.flip(np.array(cummulative))


import matplotlib.pyplot as plt

plt.figure(dpi=1200)

interpret_a = np.array(interpretability)[ids]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

s = [0.5 for n in range(len(sorted_scores))]
plt.scatter(interpret_a, sorted_scores, c='blue', s=s)
plt.plot(interpret_a, cum_flip, c='red')

plt.xlabel(r'\textbf{Interpretability} (from low (left) to high (right))', fontsize=16)
plt.ylabel(r'\textbf{Accuracy} (Micro AUC)', fontsize=16)
plt.xlim(0,1)
plt.ylim(0,1)

#plt.axvline(x=local[-1], color='r')

#plt.show()
plt.savefig('/tmp/tex_demo')

#to latex

format_number = '{:.5f}'

latex_score = '\\addplot+[only marks] coordinates{'
for i in range(len(sorted_scores)):
    if sorted_scores[i] >= 0.0:
        latex_score += '(' + format_number.format(interpret_a[i]) + "," + format_number.format(sorted_scores[i]) + ") "

latex_score += '};'


latex_cover = '\\addplot+[mark=none]'
for i in range(len(cum_flip)):
    latex_cover += '(' + format_number.format(interpret_a[i]) + "," + format_number.format(cum_flip[i]) + ")\n"
latex_cover += '};'

latex_all = '''\\begin{tikzpicture}[]
               \\begin{axis}\n'''

latex_all += latex_score #+ "\n\n" + latex_cover

latex_all += '''
\\end{axis}
\\end{tikzpicture}'''

#f1=open('/tmp/chart.tex', 'w+')
#f1.write(latex_all)
#f1.close()

