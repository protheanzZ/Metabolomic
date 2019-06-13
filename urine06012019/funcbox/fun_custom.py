import numpy as np
import numba


@numba.njit
def r2_score(y_true, y_pred):
    """
    faster r2_score11
    """
    y_pred = y_pred/y_pred.max()
    SSres = 0
    SStot = 0
    y_true_mean = y_true.mean()
    for i in range(len(y_true)):
        SSres += (y_true[i] - y_pred[i])**2
        SStot += (y_true[i] - y_true_mean)**2
    re = 1-SSres/SStot
    return re

def comb(seq):
    """
    return all possible combinations of seq
    >>>comb(['a','b','c'])
    >>>[['c', 'b', 'a'],
        ['b', 'c', 'a'],
        ['b', 'a', 'c'],
        ['c', 'a', 'b'],
        ['a', 'c', 'b'],
        ['a', 'b', 'c']]
    """
    if len(seq) == 1:
        return [seq]
    else:
        res = []
        pivot = seq[-1]
        temp = comb(seq[:-1])
        for sub_temp in temp:
            for i in range(len(sub_temp)+1):
                copy = sub_temp.copy()
                copy.insert(i, pivot)
                res.append(copy)
    return res

def comb_two(seq):
    """
    return all possible two combinations of seq
    C{len(seq), 2}
    >>>comb_two(['a','b','c','d'])
    >>>[['a', 'b'], ['a', 'c'], ['b', 'c'], 
        ['a', 'd'], ['b', 'd'], ['c', 'd']]
    """
    if len(seq) < 2:
        raise ValueError('seq must have two or more elements')
    if len(seq) == 2:
        return [seq[0], seq[1]]
    else:
        temp = comb_two(seq[:-1])
        for s in seq[:-1]:
            temp.append([s, seq[-1]])
        return temp
    
def subsets(seq):
    """
    return all subsets of one sequence
    >>>subsets
    """
    res = [[]]
    for i in range(len(seq)):
        res.extend([re+[seq[i]] for re in res])
    return res