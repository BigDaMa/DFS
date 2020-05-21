import pickle
import numpy as np


mappnames = {1:'TPE(Variance)',
			 2: 'TPE($\chi^2$)',
			 3:'TPE(FCBF)',
			 4: 'TPE(Fisher)',
			 5: 'TPE(MIM)',
			 6: 'TPE(MCFS)',
			 7: 'TPE(ReliefF)',
			 8: 'TPE(NR)',
             9: 'SA(NR)',
			 10: 'NSGA-II(NR)',
			 11: 'ES(NR)',
			 12: 'SFS(NR)',
			 13: 'SBS(NR)',
			 14: 'SFFS(NR)',
			 15: 'SBFS(NR)',
			 16: 'RFE(LR)',
			 17: 'Complete Set'
			 }






map_rows2strategy_mean_time = {}

for number_observations in [100, 1000, 10000, 100000, 1000000]:

	dataset = {}
	dataset['best_strategy'] = []
	dataset['validation_satisfied'] = []

	dataset['success_value'] = []
	dataset['success_value_validation'] = []
	dataset['times_value'] = []
	dataset['max_search_time'] = []

	dataset['distance_to_test_constraint'] = []


	def load_pickle(fname):
		data = []
		with open(fname, "rb") as f:
			while True:
				try:
					data.append(pickle.load(f))
				except EOFError:
					break
		return data


	def is_successfull_validation_and_test(exp_results):
		return len(exp_results) > 0 and 'success_test' in exp_results[-1] and exp_results[-1][
			'success_test'] == True  # also on test satisfied


	def is_successfull_validation(exp_results):
		return len(exp_results) > 0 and 'Validation_Satisfied' in exp_results[
			-1]  # constraints were satisfied on validation set



	strategy_distance_test = {}
	strategy_distance_validation = {}
	for s in range(1, len(mappnames) + 1):
		strategy_distance_test[s] = []
		strategy_distance_validation[s] = []

	run_count = 0
	for rfolder_i in range(20):
		rfolder = '/home/felix/phd/versions_dfs/scaling/experiment_observations' + str(number_observations) + '/run' + str(rfolder_i) + '/'
		try:
			info_dict = pickle.load(open(rfolder + 'run_info.pickle', "rb"))
			run_strategies_success_test = {}
			run_strategies_times = {}
			run_strategies_success_validation = {}

			validation_satisfied_by_any_strategy = False

			min_time = np.inf
			best_strategy = 0
			for s in range(1, len(mappnames) + 1):
				exp_results = []
				try:
					exp_results = load_pickle(rfolder + 'strategy' + str(s) + '.pickle')
				except:
					pass
				if is_successfull_validation_and_test(exp_results):
					runtime = exp_results[-1]['final_time']
					if runtime < min_time:
						min_time = runtime
						best_strategy = s

					run_strategies_success_test[s] = True
					run_strategies_times[s] = runtime
				else:
					run_strategies_success_test[s] = False
				# run_strategies_times[s] = runtime

				run_strategies_success_validation[s] = is_successfull_validation(exp_results)
				if run_strategies_success_validation[s]:
					validation_satisfied_by_any_strategy = True

			dataset['success_value'].append(run_strategies_success_test)
			dataset['success_value_validation'].append(run_strategies_success_validation)
			dataset['best_strategy'].append(best_strategy)
			dataset['times_value'].append(run_strategies_times)
			dataset['validation_satisfied'].append(validation_satisfied_by_any_strategy)

			run_count += 1
		except FileNotFoundError:
			pass


	strategy_time = np.zeros(len(mappnames), dtype=np.float)
	for s in range(1, len(mappnames) + 1):
		current_time = []
		for run in range(len(dataset['best_strategy'])):
			if dataset['success_value'][run][s] == True:
				current_time.append(dataset['times_value'][run][s])
			else:
				current_time.append(3 * 60 * 60)

		strategy_time[s-1] = np.mean(current_time)

	map_rows2strategy_mean_time[number_observations] = strategy_time


my_latex = ""
for s in np.array([2,12,9,16,3,10,5,13,4,7]) - 1:
#for s in np.array([17, 11, 12, 13, 14, 15, 16, 4, 7, 5, 3, 6, 1, 2, 8, 9, 10]) - 1:
	my_latex += '\\addplot+ coordinates{'
	for number_observations in [100, 1000, 10000, 100000, 1000000]:
		my_latex += '(' + str(number_observations) + ',' + str(map_rows2strategy_mean_time[number_observations][s]) + ') '
	my_latex += "};\n\\addlegendentry{"+ str(mappnames[s+1]) +"}\n\n"

print(my_latex)



