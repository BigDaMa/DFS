import pickle
from fastfeature.plotting.plotter import cool_plotting
from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature

from graphviz import Digraph
import tempfile

#file = "/tmp/chart.p"
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_23_11.p'
#file = '/home/felix/phd/fastfeature_logs/charts/chart_all_fold20_no_hyper_opt_32min.p'
file = '/home/felix/phd/fastfeature_logs/newest_28_11/chart_hyper_10_all.p'
names_all_file = '/home/felix/phd/fastfeature_logs/newest_28_11/name2id.p'



all_data = pickle.load(open(file, "rb"))
names_all: Dict[str, int]  = pickle.load(open(names_all_file, "rb"))




all_candidates: List[CandidateFeature] = pickle.load(open('/home/felix/phd/fastfeature_logs/newest_28_11/all_candidates.p', "rb"))


print(all_candidates[-1])
print(all_data['new_scores'][names_all[all_candidates[-1].get_name()]])


def candidate_to_graph(candidate: CandidateFeature, names2id, scores, starter_score, dot=Digraph(comment='Candidate'), current_id='0'):
    current_score = -1

    if isinstance(candidate, RawFeature):
        dot.node(current_id, candidate.get_name())
    else:
        current_score = scores[names2id[candidate.get_name()]]

        if starter_score < current_score:
            dot.node(current_id, candidate.transformation.name + ': ' + str(current_score)[0:4], color='green')
        else:
            dot.node(current_id, candidate.transformation.name + ': ' + str(current_score)[0:4], color='red')

        for i in range(len(candidate.parents)):
            child_id = current_id + '_' + str(i)
            dot, child_score = candidate_to_graph(candidate.parents[i], names2id, scores, starter_score, dot, child_id)

            if child_score == -1:
                dot.edge(child_id, current_id, color='blue')
            elif child_score < current_score:
                dot.edge(child_id, current_id, color='green')
            else:
                dot.edge(child_id, current_id, color='red')

    return dot, current_score


#starter_score = all_data['start_score']
#dot, _ = candidate_to_graph(all_candidates[-1], names_all, all_data['new_scores'], starter_score)
#dot.render(tempfile.mktemp('.gv'), view=False)



# calculate probability that a new transaction improves accuracy if parents already are bad
for i in range(len(all_candidates)):
    name = all_candidates[i].get_name()
    score = all_data['new_scores'][names_all[name]]
    for p in range(len(all_candidates[i].parents)):
        parent_name = all_candidates[i].parents[p].get_name()
        try:
            parent_score = all_data['new_scores'][names_all[parent_name]]
        except:
            pass