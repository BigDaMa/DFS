import numpy as np

arr = np.array([[1,2,3],
                [4,2,4],
                [5,3,4],
                [2,2,3],
                ])


import numpy_indexed as npi


def sum(grouped: npi.GroupBy, values, axis=0, dtype=None):
    """compute the sum over each group

    Parameters
    ----------
    values : array_like, [keys, ...]
        values to sum per group
    axis : int, optional
        alternative reduction axis for values
    dtype : output dtype

    Returns
    -------
    unique: ndarray, [groups]
        unique keys
    reduced : ndarray, [groups, ...]
        value array, reduced over groups
    """
    values = np.asarray(values)
    return grouped.unique, my_reduce(grouped, values, axis=axis, dtype=dtype, operator=np.add)


def size(grouped: npi.GroupBy, values, axis=0, dtype=None):
    return grouped.unique, grouped.index.count

def my_reduce(grouped: npi.GroupBy, values, operator=np.add, axis=0, dtype=None):
    """Reduce the values over identical key groups, using the given ufunc
    reduction is over the first axis, which should have elements corresponding to the keys
    all other axes are treated indepenently for the sake of this reduction

    Parameters
    ----------
    values : ndarray, [keys, ...]
        values to perform reduction over
    operator : numpy.ufunc
        a numpy ufunc, such as np.add or np.sum
    axis : int, optional
        the axis to reduce over
    dtype : output dtype

    Returns
    -------
    ndarray, [groups, ...]
    values reduced by operator over the key-groups
    """
    print(values.shape)
    values = np.take(values, grouped.index.sorter, axis=axis)
    print(values)
    print(grouped.index.start)
    print(grouped.index.shape)
    print(grouped.index.count)
    return operator.reduceat(values, grouped.index.start, axis=axis, dtype=dtype)



np.add

method = npi.GroupBy.sum

grouped = npi.group_by(arr[:, [1,2]])

print(size(grouped, arr[:,0]))
