import matplotlib.pyplot as plt
import numpy as np
import pickle

def extract_data(history, name):
	data = {}
	data['name'] = name
	data['cv_scores'] = [0.0]
	data['global_times'] = [0]
	data['complexities'] = [0]
	data['gender_unfairness'] = [0]
	for h in history:
		if h['auc'] > data['cv_scores'][-1]:
			data['cv_scores'].append(h['auc'])
			data['global_times'].append(h['global_time'])
			data['complexities'].append(h['complexity'])
			data['gender_unfairness'].append(h['fair'])
	return data

l1_feature_selection_history = pickle.load(open("/tmp/l1_feature_selection.p", "rb"))
evolutionary_selection_history = pickle.load(open("/tmp/evoltionary_feature_selection.p", "rb"))
tree_selection_history = pickle.load(open("/tmp/tree_feature_selection.p", "rb"))
recursive_selection_history = pickle.load(open("/tmp/recursive_feature_selection1.p", "rb"))

histories = []
histories.append(extract_data(l1_feature_selection_history, 'l1 regularization'))
histories.append(extract_data(evolutionary_selection_history, 'evoltionary'))
histories.append(extract_data(tree_selection_history, 'xgboost'))
histories.append(extract_data(recursive_selection_history, 'recursive'))

plt.figure()

plt.subplot(221)
for h in histories:
	plt.plot(h['global_times'], h['cv_scores'], label=h['name'])
plt.ylabel('cv score')
plt.xlabel('time (seconds)')
plt.title('cv score')

plt.subplot(222)
for h in histories:
	plt.plot(h['global_times'], h['complexities'], label=h['name'])
plt.ylabel('complexity')
plt.xlabel('time (seconds)')
plt.title('complexity')

plt.subplot(223)
for h in histories:
	plt.plot(h['global_times'], h['gender_unfairness'], label=h['name'])
plt.ylabel('gender unfairness')
plt.xlabel('time (seconds)')
plt.title('gender_unfairness')

plt.legend()

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.55,
                    wspace=0.35)

plt.show()