import numpy_indexed as npi

def count(grouped: npi.GroupBy, values, axis=0, dtype=None):
    return grouped.unique, grouped.index.count