from matplotlib import pyplot
import os
import pickle

def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

#path = '/home/felix/phd/fastfeatures_graph_search_strategies'
path = '/home/felix/phd/fastfeatures_graph_search_strategies/transfusion'
#path = '/home/felix/phd/fastfeatures_graph_search_strategies/credit'
dirs = get_immediate_subdirectories(path)

print(dirs)

pyplot.figure()

# sp1
pyplot.subplot(121)

for d in dirs:
	cross_val_accuracy = pickle.load(open(path +'/' + d + '/cross_val_accuracy.p', 'rb'))
	runtime = pickle.load(open(path +'/' + d +'/runtime.p', 'rb'))
	pyplot.plot(runtime, cross_val_accuracy, label=d)

pyplot.ylabel('accuracy')
pyplot.xlabel('sequential runtime')
pyplot.ylim((0.0, 1.0))
pyplot.title('Cross-validation F1-score')
pyplot.legend()

# sp2
pyplot.subplot(122)
for d in dirs:
	test_accuracy = pickle.load(open(path +'/' + d +'/test_accuracy.p', 'rb'))
	runtime = pickle.load(open(path +'/' + d +'/runtime.p', 'rb'))
	pyplot.plot(runtime, test_accuracy, label=d)

pyplot.ylabel('accuracy')
pyplot.xlabel('sequential runtime')
pyplot.ylim((0.0, 1.0))
pyplot.title('Test F1-score')
pyplot.legend()

pyplot.show()