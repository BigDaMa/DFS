from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np

def filter_best_combinations(all_candidates: List[CandidateFeature], all_data, candidate_id_to_stored_id, names_all):
    raw_feature_combination_candidate = {}
    raw_feature_combination_candidate_interpretability = {}
    raw_feature_combination_candidate_score = {}

    for c_i in range(len(all_candidates)):

        name = all_candidates[c_i].get_name()
        new_score = all_data['new_scores'][candidate_id_to_stored_id[names_all[name]]]
        interpretability = all_data['interpretability'][candidate_id_to_stored_id[names_all[name]]]
        #interpretability = best_candidates[c_i].get_number_of_transformations()


        raw_attributes = frozenset(all_candidates[c_i].get_raw_attributes())
        if not raw_attributes in raw_feature_combination_candidate:
            raw_feature_combination_candidate[raw_attributes] = []
            raw_feature_combination_candidate_interpretability[raw_attributes] = []
            raw_feature_combination_candidate_score[raw_attributes] = []

        raw_feature_combination_candidate[raw_attributes].append(all_candidates[c_i])
        raw_feature_combination_candidate_interpretability[raw_attributes].append(interpretability)
        raw_feature_combination_candidate_score[raw_attributes].append(new_score)

    not_pruned_candidates: List[CandidateFeature] = []
    pruned_candidates: List[CandidateFeature] = []
    for t in raw_feature_combination_candidate.items():
        #most complex representation first
        order = np.argsort(np.array(raw_feature_combination_candidate_interpretability[t[0]]))

        sorted_candidates = np.array(raw_feature_combination_candidate      [t[0]])[order]
        sorted_score      = np.array(raw_feature_combination_candidate_score[t[0]])[order]
        sorted_interpretability = np.array(raw_feature_combination_candidate_interpretability[t[0]])[order]

        for a in range(len(order)):
            should_be_pruned = False
            for b in range(a+1, len(order)):
                if sorted_score[a] <= sorted_score[b] \
                        and sorted_interpretability[a] != sorted_interpretability[b]: # if higher complex has a score smaller or equal to one representation with lower complexity
                    should_be_pruned = True
                    break

            if not should_be_pruned:
                not_pruned_candidates.append(sorted_candidates[a])
            else:
                pruned_candidates.append(sorted_candidates[a])

    return not_pruned_candidates, pruned_candidates
