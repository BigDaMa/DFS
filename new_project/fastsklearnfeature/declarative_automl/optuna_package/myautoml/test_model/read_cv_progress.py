import matplotlib.pyplot as plt
import numpy as np
import pickle

cv = np.array(pickle.load(open('/home/felix/phd2/picture_progress/al_only/felix_cv_compare.p', "rb")))
plt.plot(range(len(cv)), cv)
plt.ylim(0, 1)
plt.axhline(y=np.nanmax(cv))
plt.title('comparison prediction')
plt.ylabel('Cross-Validation R2 score')
plt.xlabel('Active Learning Iterations')
plt.show()
print(cv)

cv = np.array(pickle.load(open('/home/felix/phd2/picture_progress/al_only/felix_cv_success.p', "rb")))
plt.plot(range(len(cv)), cv)
plt.axhline(y=np.nanmax(cv))
plt.ylim(0, 1)
plt.title('success prediction')
plt.ylabel('Cross-Validation R2 score')
plt.xlabel('Active Learning Iterations')
plt.show()
print(cv)