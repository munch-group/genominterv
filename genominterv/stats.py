import pandas as pd
import numpy as np

import bisect
from statsmodels.distributions.empirical_distribution import ECDF

from collections import namedtuple

from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

from .remapping import interval_distance
from .intervals import interval_intersect, interval_union


def proximity_test(query: pd.DataFrame, annot: pd.DataFrame, samples: int=100000, 
                   npoints: int=1000, overlap_as_zero:bool=False, span_as_zero:bool=False, two_sided: bool=False) -> namedtuple:
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
    overlap_as_zero : 
        Set distance to zero if one end of a query segment overlaps an annotation segment, by default False.
        This does not apply to query segments embedded in or spanning on or more annotation segments.
    span_as_zero : 
        Set distance to zero if a query segment spans a single annotation segment, by default False.        

    Returns
    -------
    : 
        A named tuple with the test statistic and p-value.
    """

    remapped_df = interval_distance(query, annot, 
                                    overlap_as_zero=overlap_as_zero,
                                    span_as_zero=span_as_zero)
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

