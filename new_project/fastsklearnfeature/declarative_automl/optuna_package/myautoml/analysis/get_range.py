import pickle
import matplotlib.pyplot as plt
import numpy as np

new_dict = pickle.load(open("/home/felix/phd2/picture_progress/data.p", "rb"))

plt.hist(new_dict['training_times'], bins=1000)
print(len(new_dict['training_times']))
print('min: ' + str(np.min(new_dict['training_times'])) + ' max: ' + str(np.max(new_dict['training_times'])) + ' median: ' + str(np.median(new_dict['training_times'])))
plt.show()

plt.hist(new_dict['inference_times'], bins=1000)
print(len(new_dict['inference_times']))
print('min: ' + str(np.min(new_dict['inference_times'])) + ' max: ' + str(np.max(new_dict['inference_times'])) + ' median: ' + str(np.median(new_dict['inference_times'])))
plt.show()

plt.hist(new_dict['pipeline_sizes'], bins=1000)
print(len(new_dict['pipeline_sizes']))
print('min: ' + str(np.min(new_dict['pipeline_sizes'])) + ' max: ' + str(np.max(new_dict['pipeline_sizes'])) + ' median: ' + str(np.median(new_dict['pipeline_sizes'])))
print(new_dict['pipeline_sizes'])
plt.show()