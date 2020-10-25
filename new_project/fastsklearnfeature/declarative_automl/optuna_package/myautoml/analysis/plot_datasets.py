import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_step1.p', 'rb'))
new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results.p', 'rb'))

for k, v in new_dct.items():
    print(k)

    name = openml.datasets.get_dataset(dataset_id=k).name

    static = v['static']
    dynamic = v['dynamic']

    plt.errorbar(np.array(range(len(static))) + 1, np.mean(static, axis=1), yerr=np.std(static, axis=1), color='red',
                 label='static')
    plt.errorbar(np.array(range(len(dynamic))) + 1, np.mean(dynamic, axis=1), yerr=np.std(dynamic, axis=1),
                 color='blue', label='dynamic')
    plt.ylabel('F1 Score')
    plt.xlabel('Search time constraint (Minutes)')
    plt.ylim((0, 1))
    plt.title(name)
    plt.legend()

    plt.savefig('/home/felix/phd2/picture_progress/all_test_datasets/img/data_' + name + '_' + str(k) + '.png')
    plt.clf()

    print('data: ' + name + ' dynamic: ' + "{:.2f}".format(np.mean(v['dynamic'])) + ' +- ' + "{:.2f}".format(np.std(v['dynamic'])) + ' static: ' + "{:.2f}".format(np.mean(v['static'])) + ' +- ' + "{:.2f}".format(np.std(v['static'])))