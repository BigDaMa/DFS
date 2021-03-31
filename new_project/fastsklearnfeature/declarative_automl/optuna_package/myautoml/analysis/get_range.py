import pickle
import matplotlib.pyplot as plt
import numpy as np

new_dict = pickle.load(open("/home/felix/phd2/picture_progress/data.p", "rb"))

plt.hist(new_dict['training_times'], bins=1000)
print(len(new_dict['training_times']))
print('min: ' + str(np.min(new_dict['training_times'])) + ' max: ' + str(np.max(new_dict['training_times'])) + ' median: ' + str(np.median(new_dict['training_times'])))
print("1 perc: " + str(np.percentile(new_dict['training_times'], 1)))
print("2 perc: " + str(np.percentile(new_dict['training_times'], 2)))
print("4 perc: " + str(np.percentile(new_dict['training_times'], 4)))
print("8 perc: " + str(np.percentile(new_dict['training_times'], 8)))
print("16 perc: " + str(np.percentile(new_dict['training_times'], 16)))
print("32 perc: " + str(np.percentile(new_dict['training_times'], 32)))

plt.show()

plt.hist(new_dict['inference_times'], bins=1000)
print(len(new_dict['inference_times']))
print('min: ' + str(np.min(new_dict['inference_times'])) + ' max: ' + str(np.max(new_dict['inference_times'])) + ' median: ' + str(np.median(new_dict['inference_times'])))
print("1 perc: " + str(np.percentile(new_dict['inference_times'], 1)))
print("2 perc: " + str(np.percentile(new_dict['inference_times'], 2)))
print("4 perc: " + str(np.percentile(new_dict['inference_times'], 4)))
print("8 perc: " + str(np.percentile(new_dict['inference_times'], 8)))
print("16 perc: " + str(np.percentile(new_dict['inference_times'], 16)))
print("32 perc: " + str(np.percentile(new_dict['inference_times'], 32)))

plt.show()

plt.hist(new_dict['pipeline_sizes'], bins=1000)
print(len(new_dict['pipeline_sizes']))
print('min: ' + str(np.min(new_dict['pipeline_sizes'])) + ' max: ' + str(np.max(new_dict['pipeline_sizes'])) + ' median: ' + str(np.median(new_dict['pipeline_sizes'])))
print("1 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 1)))
print("2 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 2)))
print("4 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 4)))
print("8 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 8)))
print("16 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 16)))
print("32 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 32)))
print("50 perc: " + str(np.percentile(new_dict['pipeline_sizes'], 50)))
print(new_dict['pipeline_sizes'])
plt.show()