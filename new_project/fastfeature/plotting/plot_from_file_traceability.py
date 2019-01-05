import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import Dict

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'

t_file = '/home/felix/phd/fastfeature_logs/charts/traceability.p'
names_trace_file = '/home/felix/phd/fastfeature_logs/traceability/name2id.p'
names_all_file = '/home/felix/phd/fastfeature_logs/newest_28_11/name2id.p'


t_all_data = pickle.load(open(t_file, "rb"))
all_data = pickle.load(open(file, "rb"))

names_trace: Dict[str, int]  = pickle.load(open(names_trace_file, "rb"))
names_all: Dict[str, int]  = pickle.load(open(names_all_file, "rb"))

# check names fit each other
for (t_name, t_id) in names_trace.items():
    assert names_all[t_name] == t_id, "names are not the same"



print(len(t_all_data['traceability']))
print(len(all_data['new_scores']))

cool_plotting(t_all_data['traceability'],
              all_data['new_scores'],
              all_data['names'],
              all_data['start_score'],
              (0.78, 0.88))