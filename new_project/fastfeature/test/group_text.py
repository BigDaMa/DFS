import numpy as np
from typing import Dict, Any, List

def fit(data, keys, method):
    mapping: Dict[Any, float] = {}
    for record_i in range(data.shape[0]):
        key = tuple(element for element in data[record_i, keys])
        if not key in mapping:
            mapping[key]: List[float] = []
        mapping[key].append(float(data[record_i, 0]))

    final_mapping = {}
    for k, v in mapping.items():
        final_mapping[k] = method(np.array(v))

    return final_mapping


arr = np.array([[1,2,'a'],
                [2,2,'a'],
                [2,2,'b'],
                [3,3,'b'],
                ])


print(fit(arr,[1,2], np.sum))