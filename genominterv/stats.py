import pandas as pd
import numpy as np

import bisect
from statsmodels.distributions.empirical_distribution import ECDF

from collections import namedtuple

from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

from .intervals import interval_relative_distance, interval_intersect, interval_union


def proximity_test(query: pd.DataFrame, annot: pd.DataFrame, samples: int=10000, npoints: int=1000, overlap_as_zero=False, two_sided: bool=False) -> namedtuple:
    """
    Test for proximity of intervals to a set of annotations.

    Parameters
    ----------
    query :
        Data frame with query intervals.
    annot :
        Data frame with annotation intervals.
    samples : 
        Number of bootstrap samples to use.
    npoints :
        Number of points to use in the ECDF.
    two_sided :
        Whether to test for proximity in both directions.
        
    Returns
    -------
    : 
        A named tuple with the test statistic and p-value.
    """

    # annot = interval_collapse(annot)

    # if overlap_as_zero:
    #     overlapping_query_idx = []
    #     for i, q in enumerate(query):
    #         for a in annot:
    #             if q[0] < a[1] and q[1] > a[0]:
    #                 overlapping_query_idx.append(i)
    #                 break

    # query = interval_diff(query, annot)
    # remapped_df = interval_relative_distance(query, annot, overlap_as_zero=overlap_as_zero)

    remapped_df = interval_relative_distance(query, annot)
    distances = abs(remapped_df.start)

    def _stat(distances, npoints):
        obs_ecdf = ECDF(distances)
        points = np.linspace(0, 0.5, num=npoints)    
        test_stat = sum(obs_ecdf(points) - 2 * points) * 2 / npoints
        return test_stat
    
    test_stat = _stat(distances, npoints)
    
    null_distr = list()
    for i in range(samples):
        sampled_distances = np.random.uniform(0, 0.5, len(distances))
        # we compute absolute values for the null distribution
        null_distr.append(_stat(sampled_distances, npoints))
    null_distr.sort()
    
    if two_sided:
        p_value = (len(null_distr) - bisect.bisect_left(list(map(abs, null_distr)), abs(test_stat))) / len(null_distr)
    else:
        p_value = (len(null_distr) - bisect.bisect_left(null_distr, test_stat)) / len(null_distr)
    
    TestResult = namedtuple('TestResult', ['statistic', 'pvalue'])
    return TestResult(test_stat, p_value)


def jaccard_stat(a: List[tuple], b:List[tuple]) -> float:
    """
    Compute Jaccard overlap test statistic.

    Parameters
    ----------
    a :
        List of tuples with (start, end) coordinates.
    a :
        List of tuples with (start, end) coordinates.
        
    Returns
    -------
    : 
        The Jaccard test statistic.
    """    

    inter = interval_intersect(a, b)
    union = interval_union(a, b)

    return sum(inter.end - inter.start) / sum(union.end - union.start)

