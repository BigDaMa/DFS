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


latex = ''

dynamic_means = []
static_means = []

def tr(number):
    return float("{:.2f}".format(number))

for k, v in new_dct.items():
    print(k)
    name = openml.datasets.get_dataset(dataset_id=k, download_data=False).name
    print('data: ' + name)

    print(v)

    latex += str(k) + ' & '

    print(latex)

    #x_axis = [0.01, 0.05, 0.1, 1.0, 10.0]
    #x_axis = [0.00074, 0.00078, 0.00085, 0.00143, 0.00261, 0.00664]
    x_axis = np.array(range(len(v['dynamic']))) + 1
    x_axis_name = 'Training time constraint (Seconds)'
    #x_axis_name = 'Search time constraint (Minutes)'
    #x_axis_name = 'Inference time constraint (Seconds)'


    try:
        if 'static' in v:
            static = v['static']
            print(' static: ' + "{:.2f}".format(np.mean(v['static'])) + ' +- ' + "{:.2f}".format(np.std(v['static'])))

            if tr(np.mean(v['static'])) >= tr(np.mean(v['dynamic'])):
                latex += "$\\textbf{" + "{:.2f}".format(np.mean(v['static'])) + '} \pm ' + "{:.2f}".format(np.std(v['static'])) + "$" + ' &'
            else:
                latex += "$" + "{:.2f}".format(np.mean(v['static'])) + ' \pm ' + "{:.2f}".format(
                    np.std(v['static'])) + "$" + ' &'

            static_means.append(np.mean(v['static']))

        dynamic = v['dynamic']

        print( ' dynamic: ' + "{:.2f}".format(np.mean(v['dynamic'])) + ' +- ' + "{:.2f}".format(np.std(v['dynamic'])))

        if tr(np.mean(v['static'])) > tr(np.mean(v['dynamic'])):
            latex += "$" + "{:.2f}".format(np.mean(v['dynamic'])) + ' \pm ' + "{:.2f}".format(np.std(v['dynamic'])) + "$"
        else:
            latex += "$\\textbf{" + "{:.2f}".format(np.mean(v['dynamic'])) + '} \pm ' + "{:.2f}".format(np.std(v['dynamic'])) + "$"

        dynamic_means.append(np.mean(v['dynamic']))

        latex += ' \\\\ \n'
        print(latex)
    except:
        pass

latex += '\midrule \n'
latex += 'Mean' + ' & '
latex += "$" + "{:.2f}".format(np.mean(static_means)) + ' \pm ' + "{:.2f}".format(np.std(static_means)) + "$" + ' &'
latex += "$\\textbf{" + "{:.2f}".format(np.mean(dynamic_means)) + '} \pm ' + "{:.2f}".format(np.std(dynamic_means)) + "$"
latex += ' \\\\ \n'

print(latex)