import numpy as np
from scipy import stats
import pandas as pd

column_data = [0.1, 0.2, 0.0, -1000.1]
#print np.matrix(stats.zscore(column_data)).T
#print np.matrix(stats.mstats.winsorize(column_data)).T
#print np.matrix(stats.mstats.rsh(column_data)).T
#print np.matrix(stats.mstats.trimtail(column_data)).T
#print np.matrix(stats.mstats.rankdata(column_data)).T
print np.matrix(stats.mstats.plotting_positions(column_data)).T




