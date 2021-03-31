import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_step1.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_training_time_constraint.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_inference_time_constraint.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_only_success.p', 'rb'))

new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_pipeline_size_constraint.p', 'rb'))
#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_memory_size_constraint.p', 'rb'))

for k, v in new_dct.items():
    print(k)
    name = openml.datasets.get_dataset(dataset_id=k, download_data=False).name
    print('data: ' + name)

    print(v)

    #x_axis = [0.01, 0.05, 0.1, 1.0, 10.0]
    #x_axis = [0.00074, 0.00078, 0.00085, 0.00143, 0.00261, 0.00664]
    x_axis = np.array(range(len(v['dynamic']))) + 1
    x_axis_name = 'Training time constraint (Seconds)'
    #x_axis_name = 'Search time constraint (Minutes)'
    #x_axis_name = 'Inference time constraint (Seconds)'


    if 'static' in v:
        static = v['static']
        plt.errorbar(x_axis, np.mean(static, axis=1), yerr=np.std(static, axis=1),
                     color='red',
                     label='static')

        print(' static: ' + "{:.2f}".format(np.mean(v['static'])) + ' +- ' + "{:.2f}".format(np.std(v['static'])))

    dynamic = v['dynamic']

    plt.errorbar(x_axis, np.mean(dynamic, axis=1), yerr=np.std(dynamic, axis=1),
                 color='blue', label='dynamic')
    plt.ylabel('F1 Score')
    plt.xlabel(x_axis_name)
    plt.ylim((0, 1))
    plt.title(name)
    #plt.xscale('log')
    plt.legend()

    plt.savefig('/home/felix/phd2/picture_progress/all_test_datasets/img/data_' + name + '_' + str(k) + '.png')
    plt.clf()

    print( ' dynamic: ' + "{:.2f}".format(np.mean(v['dynamic'])) + ' +- ' + "{:.2f}".format(np.std(v['dynamic'])))