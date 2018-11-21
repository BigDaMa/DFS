from scipy import stats

print(stats.mstats.rankdata([0,2,3,4,5,-1]))

print(stats.mstats.rankdata(['a', 'd', 'b']))