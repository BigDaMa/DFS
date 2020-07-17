import pickle
import numpy as np
import matplotlib.pyplot as plt

#trials = pickle.load(open("/home/felix/phd2/experiments_constrained_HPO/trials.p", "rb"))
study = pickle.load(open("/home/felix/phd2/experiments_optuna/10_minutes_max/optuna_study.p", "rb"))

success_time = []
success_total_time = []
success_acc = []
for i in range(len(study.trials)):
    if 'total_time' in study.trials[i].user_attrs and \
            'training_time' in study.trials[i].user_attrs and \
            study.trials[i].user_attrs['training_time'] > 0 and \
            type(None) != type(study.trials[i].value):
        success_time.append(study.trials[i].user_attrs['training_time'])
        success_total_time.append(study.trials[i].user_attrs['total_time'])
        success_acc.append(study.trials[i].value)
        print(str(i) + ' : ' + str(study.trials[i].value))
        print()


success_time = np.array(success_time)
success_acc = np.array(success_acc)
success_total_time = np.array(success_total_time)

print(success_time)

search_times = []
results = []
search_times_stopped = []

constraints = list(range(1, 60*10))

for training_time_constraint in constraints:
    mask = success_time <= training_time_constraint
    search_times.append(np.sum(success_total_time[mask]))
    results.append(np.max(success_acc[mask]))

    number_failed = (len(success_time) - np.sum(mask)) * training_time_constraint
    search_times_stopped.append(np.sum(success_total_time[mask]) + (len(success_time) - np.sum(mask)) * training_time_constraint)


plt.plot(constraints, search_times, label='Perfect Pruning')
plt.plot(constraints, search_times_stopped, label='Stopped Execution')
plt.axhline(y=np.sum(success_total_time), label='No Pruning')
plt.legend(loc='lower right')

plt.xlabel('Training Time Constraint')
plt.ylabel('Total Search Time')
plt.show()

plt.plot(constraints, results)
plt.xlabel('Training Time Constraint')
plt.ylabel('Achieved AUC')
plt.show()




