import pandas as pd
import numpy as np

from functools import reduce

from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

# In all of the following, the list of intervals must be sorted and 
# non-overlapping. We also assume that the intervals are half-open, so
# that x is in tp(start, end) iff start <= x and x < end.

def flatten(list_of_tps):
    """
    Convert a list of sorted intervals to a list of endpoints.

    Parameters
    ----------
    list_of_tps : 
        List of intervals.

    Returns
    ----------
    : 
        A list of interval ends
    """
    return reduce(lambda ls, ival: ls + list(ival), list_of_tps, [])


def unflatten(list_of_endpoints):
    """
    Convert a list of sorted endpoints into a list of intervals.

    Parameters
    ----------
    list_of_endpoints : 
        List of endpoints.

    Returns
    ----------
    : 
        List of intervals.
    """
    return [ [list_of_endpoints[i], list_of_endpoints[i + 1]]
          for i in range(0, len(list_of_endpoints) - 1, 2)]


def merge(query: list, annot: list, op: Callable) -> list:
    """
    Merge two lists of sorted (start, end) intervals according to the boolean function op.

    Parameters
    ----------
    a : 
        List of intervals.
    b : 
        List of intervals.

    Returns
    ----------
    : 
        List of intervals.
    """
    a_endpoints = flatten(query)
    b_endpoints = flatten(annot)

    # assert a_endpoints == sorted(a_endpoints), "not sorted or non-overlapping"
    # assert b_endpoints == sorted(b_endpoints), "not sorted or non-overlapping"


    sentinel = max(a_endpoints[-1], b_endpoints[-1]) + 1
    a_endpoints += [sentinel]
    b_endpoints += [sentinel]

    a_index = 0
    b_index = 0

    res = []

    scan = min(a_endpoints[0], b_endpoints[0])
    while scan < sentinel:
        in_a = not ((scan < a_endpoints[a_index]) ^ (a_index % 2))
        in_b = not ((scan < b_endpoints[b_index]) ^ (b_index % 2))
        in_res = op(in_a, in_b)

        if in_res ^ (len(res) % 2):
            res += [scan]
        if scan == a_endpoints[a_index]: 
            a_index += 1
        if scan == b_endpoints[b_index]: 
            b_index += 1
        scan = min(a_endpoints[a_index], b_endpoints[b_index])

    return unflatten(res)

def diff(a: list, b: list) -> list:
    """
    Difference intervals of two sorted lists of (start, end) intervals.

    Parameters
    ----------
    a :
        List of intervals.
    b : 
        List of intervals.

    Returns
    ----------
    :
        List of intervals.
    """
    if not (a and b):
        return a and a or b
    return merge(a, b, lambda in_a, in_b: in_a and not in_b)

def union(a: list, b: list) -> list:
    """
    Union intervals of two sorted lists of (start, end) intervals.

    Parameters
    ----------
    a :
        List of intervals.
    b :
        List of intervals.

    Returns
    ----------
    :
        List of intervals.
    """
    if not (a and b):
        return []
    return merge(a, b, lambda in_a, in_b: in_a or in_b)

def intersect(a: list, b: list) -> list:
    """
    Intersection intervals of two sorted lists of (start, end) intervals.

    Parameters
    ----------
    a :
        List of intervals.
    b :
        List of intervals.

    Returns
    ----------
    :
        List of intervals.
    """
    if not (a and b):
        return []
    return merge(a, b, lambda in_a, in_b: in_a and in_b)

def collapse(a):
    """
    Converts a list of sorted overlapping intervals to non-overlapping
    intervals spanning each inCollapsed non intervals of two sorted 
    lists of (start, end) intervals.

    Parameters
    ----------
    a :
        List of intervals.

    Returns
    ----------
    :
        List of intervals.
    """    
    a_union = [list(a[0])]
    for i in range(1, len(a)):
        x = a[i]
        if a_union[-1][1] < x[0]:
            a_union.append(list(x))
        else:
            a_union[-1][1] = x[1]
    return a_union

def invert(a: list, left: int, right: int) -> list:
    """
    Produces the complement of a list of sorted intervals 
    limited by the left `left` and `right` parameters.

    Parameters
    ----------
    a :
        List of intervals.
    left :
        Left boundary position.
    right :
        Left boundary position.

    Returns
    ----------
    :
        List of intervals.
    """
    starts, ends = zip(*collapse(sorted(a)))
    
    assert left <= starts[0] and right >= ends[-1]    

    starts = list(starts)
    ends = list(ends)
        
    ends.insert(0, left)
    starts.append(right)

    # remove first and last interval if they are empty
    if starts[0] == ends[0]:
        del starts[0]
        del ends[0]
    if starts[-1] == ends[-1]: 
        del starts[-1]
        del ends[-1]            
    inverted = zip(ends, starts)
    inverted = list(map(tuple, inverted))
    return inverted
