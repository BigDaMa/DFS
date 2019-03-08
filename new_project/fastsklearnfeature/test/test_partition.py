from typing import List, Dict, Set

def get_length_2_partition(cost: int) -> List[List[int]]:
    partition: List[List[int]] = []

    p = cost - 1
    while p >= cost - p:
        partition.append([p, cost - p])
        p = p - 1
    return partition


print(get_length_2_partition(6))
