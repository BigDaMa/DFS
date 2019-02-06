from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np


def enrich_candidates(candidates: List[CandidateFeature], all_data, candidate_id_to_stored_id, names_all):
    interpretabilities = []
    scores = []
    names = []
    for c in candidates:
        name = c.get_name()
        score = all_data['new_scores'][candidate_id_to_stored_id[names_all[name]]]

        interpretabilities.append(all_data['interpretability'][candidate_id_to_stored_id[names_all[name]]])
        scores.append(score)
        names.append(name)

    return names, scores, interpretabilities

def enrich_candidates_all(not_pruned_candidates: List[CandidateFeature], pruned_candidates: List[CandidateFeature], all_data, candidate_id_to_stored_id, names_all):
    names_pruned, scores_pruned, interpretabilities_pruned = enrich_candidates(pruned_candidates, all_data, candidate_id_to_stored_id, names_all)
    names_not_pruned, scores_not_pruned, interpretabilities_not_pruned = enrich_candidates(not_pruned_candidates, all_data, candidate_id_to_stored_id,
                                                                names_all)

    return names_not_pruned, scores_not_pruned, interpretabilities_not_pruned, \
           names_pruned, scores_pruned, interpretabilities_pruned

