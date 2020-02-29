import matplotlib.pyplot as plt
import numpy as np
means_runs = [787.0493352866757,  718.761317790577, 634.2834941557546, 250.15510002030265]
stds_runs = [1110.089044283395, 1059.1803140282298, 979.3843788214118, 513.367069233026]

plt.errorbar(np.arange(len(means_runs)), means_runs, stds_runs, fmt='ok', lw=3)
plt.ylim(bottom=0)
plt.xticks(np.arange(4), ('Single-objective', 'Multi-objective', 'meta-learning', 'oracle'))
plt.show()