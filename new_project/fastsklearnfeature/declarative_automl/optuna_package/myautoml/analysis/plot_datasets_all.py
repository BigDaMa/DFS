import matplotlib.pyplot as plt
import numpy as np
import pickle
import openml

new_dct_success = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_only_success.p', 'rb'))
new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn_best.p', 'rb'))
new_dct_vanilla = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_auto_sklearn_vanilla.p', 'rb'))

#new_dct = pickle.load(open('/home/felix/phd2/picture_progress/all_test_datasets/all_results_only_success.p', 'rb'))

for k, v in new_dct_success.items():
    print(k)
    name = openml.datasets.get_dataset(dataset_id=k, download_data=False).name
    print('data: ' + name)

    '''
    if 'static' in v:
        static = v['static']
        plt.errorbar(np.array(range(len(static))) + 1, np.mean(static, axis=1), yerr=np.std(static, axis=1),
                     color='red',
                     label='static')

        print(' static: ' + "{:.2f}".format(np.mean(v['static'])) + ' +- ' + "{:.2f}".format(np.std(v['static'])))
    '''

    dynamic = v['dynamic']
    plt.errorbar(np.array(range(len(dynamic))) + 1, np.mean(dynamic, axis=1), yerr=np.std(dynamic, axis=1),
                 color='blue', label='dynamic')

    autosklearn = new_dct[k]['dynamic']
    plt.errorbar(np.array(range(len(autosklearn))) + 1, np.mean(autosklearn, axis=1), yerr=np.std(autosklearn, axis=1),
                 color='green', label='autosklearn metalearning and ensembling')

    autosklearn_vanilla = new_dct_vanilla[k]['dynamic']
    plt.errorbar(np.array(range(len(autosklearn_vanilla))) + 1, np.mean(autosklearn_vanilla, axis=1), yerr=np.std(autosklearn_vanilla, axis=1),
                 color='pink', label='autosklearn vanilla')


    plt.ylabel('F1 Score')
    plt.xlabel('Search time constraint (Minutes)')
    plt.ylim((0, 1))
    plt.title(name)
    plt.legend()

    plt.savefig('/home/felix/phd2/picture_progress/all_test_datasets/img_all/data_' + name + '_' + str(k) + '.png')
    plt.clf()

    print(' dynamic: ' + "{:.2f}".format(np.mean(v['dynamic'])) + ' +- ' + "{:.2f}".format(np.std(v['dynamic'])))