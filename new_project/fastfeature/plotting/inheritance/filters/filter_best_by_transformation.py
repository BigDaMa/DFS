from typing import Dict, List
from fastfeature.candidates.CandidateFeature import CandidateFeature
from fastfeature.candidates.RawFeature import RawFeature
import numpy as np

def filter_best_accuracy_per_transformation(all_candidates: List[CandidateFeature],
                                            all_data,
                                            candidate_id_to_stored_id,
                                            names_all):
    number_transaction_max_accurate_candidate = {}
    number_transaction_max_accurate_score = {}

    for c in all_candidates:
        name = c.get_name()
        score = all_data['new_scores'][candidate_id_to_stored_id[names_all[name]]]

        t = c.get_number_of_transformations()
        if not t in number_transaction_max_accurate_score or number_transaction_max_accurate_score[t] < score:
            number_transaction_max_accurate_candidate[t] = c
            number_transaction_max_accurate_score[t] = score

    return number_transaction_max_accurate_candidate, number_transaction_max_accurate_score