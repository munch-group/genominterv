import pandas as pd
import numpy as np

import bisect
from statsmodels.distributions.empirical_distribution import ECDF

from collections import namedtuple

from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

from .remapping import interval_distance
from .intervals import interval_intersect, interval_union
from .remapping import remap_interval_data

# def proximity_test(query: pd.DataFrame, annot: pd.DataFrame, samples: int=10000, 
#                    npoints: int=1000, two_sided:bool=False, cores:int=1,
#                    overlap_as_zero:bool=False, span_as_zero:bool=False, 
#                    return_boot:bool=False) -> namedtuple:
#     """
#     Test for proximity of intervals to a set of annotations.

#     Parameters
#     ----------
#     query :
#         Data frame with query intervals.
#     annot :
#         Data frame with annotation intervals.
#     samples : 
#         Number of bootstrap samples to use.
#     npoints :
#         Number of points to use in the ECDF.
#     two_sided :
#         Whether to test for proximity in both directions.
#     cores : 
#         Number of cores to use, by default 1.
#     overlap_as_zero : 
#         Set distance to zero if one end of a query segment overlaps an 
#         annotation segment, by default False.
#         This does not apply to query segments embedded in or spanning one
#         or more annotation segments.
#     span_as_zero : 
#         Set distance to zero if a query segment spans a single annotation 
#         segment, by default False.        
#     return_boot : 
#         Also return a list of the bootstrap statistics, by default False.

#     Returns
#     -------
#     : 
#         A named tuple with the test statistic and p-value.
#     """

#     remapped_df = interval_distance(query, annot, 
#                                     relative=True,
#                                     overlap_as_zero=overlap_as_zero,
#                                     span_as_zero=span_as_zero)
#     distances = abs(remapped_df.start) 
#     # distances = np.linspace(0, 0.5, num=npoints)
    
#     def _stat(distances, npoints):
#         obs_ecdf = ECDF(distances)
#         points = np.linspace(0, 0.5, num=npoints)  
#         test_stat = sum(obs_ecdf(points) - 2 * points) * 2 / npoints
#         return test_stat

#     test_stat = _stat(distances, npoints)


#     try:
#         from multiprocess import Pool
#         multi = True
#     except ImportError:
#         multi = True

#     # compute absolute values for the null distribution
#     if cores > 1 and multi:
#         def _fun(_):
#             sampled_distances = np.random.uniform(0, 0.5, len(distances))
#             # we compute absolute values for the null distribution
#             return _stat(sampled_distances, npoints)
        
#         with Pool(cores) as pool:
#             gen = pool.map(_fun, range(samples))
#         null_distr = list(gen)
#     else:
#         if cores > 1:
#             print("multiprocess library is required for multiprocessing:",
#                   "    conda install conda-forge::multiprocess",
#                   file=sys.stderr)
#         null_distr = list()
#         for i in range(samples):
#             sampled_distances = np.random.uniform(0, 0.5, len(distances))
#             null_distr.append(_stat(sampled_distances, npoints))    
    
#     if two_sided:
#         assert 0, "two_sided NOT IMPLEMENTED YET"
#         # p_value = (len(null_distr) - bisect.bisect_left(
#         #     list(map(abs, null_distr)), abs(test_stat))) / len(null_distr)
#     else:
#         p_value = (len(null_distr) - bisect.bisect_left(
#             null_distr, test_stat)) / len(null_distr)

#     if return_boot:
#         TestResult = namedtuple('TestResult', ['statistic', 'pvalue', 'bootstraps'])
#         return TestResult(test_stat, p_value, null_distr)
#     else:
#         TestResult = namedtuple('TestResult', ['statistic', 'pvalue'])
#         return TestResult(test_stat, p_value)


def proximity_stat(query:pd.DataFrame, annot:pd.DataFrame):
    """
    Proximity test statistic. Computes the distance between query segment and the 
    closest annotation segment relative to the distance between the two annotations 
    flanking the query (distances 0-0.5). The test statistic is the mean of these.

    Parameters
    ----------
    query :
        pandas.DataFrame with query interval coordinates as start and end columns.
    annot :
        pandas.DataFrame with annotation interval coordinates as start and end columns.
        
    Returns
    -------
    : 
        The proximity test statistic.
    """   
    return 0.5 - remap_interval_data(query, annot, relative=True).start.abs().mean()


def jaccard_stat(query:pd.DataFrame, annot:pd.DataFrame) -> float:
    """
    Jaccard overlap test statistic.

    Parameters
    ----------
    query :
        pandas.DataFrame with query interval coordinates as start and end columns.
    annot :
        pandas.DataFrame with annotation interval coordinates as start and end columns.
        
    Returns
    -------
    : 
        The Jaccard test statistic.
    """    

    inter = interval_intersect(query, annot)
    union = interval_union(query, annot)

    return sum(inter.end - inter.start) / sum(union.end - union.start)


# def roc_test(a: List[tuple], b:List[tuple]) -> float:
