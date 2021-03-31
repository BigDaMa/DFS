import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_step1.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_training_time_constraint_part1.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_inference_time_constraint_part1.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_only_success.p', 'rb'))
new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_memory_size_constraint.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_pipeline_size_constraint.p', 'rb'))

latex = ''

dynamic_means = []
static_means = []

def tr(number):
    return float("{:.2f}".format(number))

#x_axis = [0.01, 0.05, 0.1, 1.0, 10.0]

map_constraint2values_static = {}
map_constraint2values_dynamic = {}

counter = 0
for k, v in new_dct.items():
    if len(v['dynamic']) >= counter:
        counter = len(v['dynamic'])
        for constraint_v in range(len(v['dynamic'])):
            if not constraint_v in map_constraint2values_static:
                map_constraint2values_static[constraint_v] = []
                map_constraint2values_dynamic[constraint_v] = []
            map_constraint2values_static[constraint_v].append(np.mean(v['static'][constraint_v]))
            map_constraint2values_dynamic[constraint_v].append(np.mean(v['dynamic'][constraint_v]))

latex += 'Static'
for constraint_v in range(counter):
    latex += " & $" + "{:.2f}".format(np.mean(map_constraint2values_static[constraint_v])) + ' \pm ' + "{:.2f}".format(np.std(map_constraint2values_static[constraint_v])) + "$"
latex += '\\\\ \n'
latex += 'Dynamic'
for constraint_v in range(counter):
    latex += " & $" + "{:.2f}".format(np.mean(map_constraint2values_dynamic[constraint_v])) + ' \pm ' + "{:.2f}".format(np.std(map_constraint2values_dynamic[constraint_v])) + "$"
latex += '\\\\ \n'

print(latex)