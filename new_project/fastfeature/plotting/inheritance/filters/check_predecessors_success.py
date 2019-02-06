from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature

def check_if_predecessors_were_more_successful_single(candidate: CandidateFeature, score, all_data, candidate_id_to_stored_id, names_all, start=False):
    name = candidate.get_name()
    new_score = all_data['new_scores'][candidate_id_to_stored_id[names_all[name]]]
    if start == True:
        score = new_score
    else:
        if new_score >= score:
            return True

    if not isinstance(candidate, RawFeature):
        for p in candidate.parents:
            if check_if_predecessors_were_more_successful_single(p, score, all_data, candidate_id_to_stored_id, names_all):
                return True
    return False


def check_if_predecessors_were_more_successful(all_candidates: List[CandidateFeature], all_data, candidate_id_to_stored_id, names_all):
    candidates_pruned: List[CandidateFeature] = []
    candidates_not_pruned: List[CandidateFeature] = []

    for c in all_candidates:
        if check_if_predecessors_were_more_successful_single(c, -1.0, all_data, candidate_id_to_stored_id, names_all, True):
            candidates_pruned.append(c)
        else:
            candidates_not_pruned.append(c)

    return candidates_not_pruned, candidates_pruned