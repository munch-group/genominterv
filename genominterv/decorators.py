import pandas as pd
import numpy as np
import bisect
import random
import sys
from functools import wraps

from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

from .chrom_sizes import chrom_sizes

def by_chrom(func: Callable) -> Callable: 
    """
    Decorator that converts a function operating on a pd.DataFrame with
    intervals from a single chromosome to one operating on one with
    many chromosomes.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        # make a local copy with reset indexes
        data_frames = [df.reset_index() for df in args]

        # get all chromosoems in arguments
        chromosomes = set()
        for df in data_frames:
            chromosomes.update(df['chrom'].unique())
        chromosomes = sorted(chromosomes)

        # get indexes (possibly none) for each chromosome in each frame
        idx = list()
        for df in data_frames:
            d = dict((chrom, []) for chrom in chromosomes)
            gr = df.groupby('chrom').groups
            d.update(gr)
            idx.append(d)

        # call func on subsets of each argument matching a chromosome
        results = list()
        for chrom in chromosomes:
            func_args = list()
            for i, df in enumerate(data_frames):
                func_args.append(df.loc[idx[i][chrom]])
            _df = func(*func_args, **kwargs)
            if _df.index.size > 0:
                results.append(_df)
        if len(results):
            # df = pd.concat(results).reset_index()
            df = pd.concat(results).reset_index(drop=True)        
            # if 'index' in df:
            #     df.drop(columns=['index'], inplace=True)        
            return df
        else:
            return _df # to get the empty columns
    return wrapper


def with_chrom(func: Callable) -> Callable:
    """
    Decorator for converting a function operating on a list of (start, end) tuples to one
    that takes a pd.DataFrame with chrom, start, end columns. Also sorts intervals.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        chrom_set = set()
        tps_list = list()
        for df in args:            
            chrom_set.update(df['chrom'])
            tps = sorted(zip(df['start'], df['end']))
            tps_list.append(tps)
        assert len(chrom_set) == 1
        chrom = chrom_set.pop()
        res_df = pd.DataFrame.from_records(func(*tps_list, **kwargs), columns = ['start', 'end'])
        res_df['chrom'] = chrom
        return res_df
    return wrapper

# A decorator that preserves the signature.
def genomic(func: Callable) -> Callable:
    """
    Decorator for converting a function operating on lists of (start, end) tuples to one
    that takes data frames with chrom, start, end columns and executes on each
    chromosome individually.

    Parameters
    ----------
    func : 
        Function accepting (start, end) tuples.

    Returns
    -------
    :
        A decorated function that takes data frames with chrom, start, end columns and executes on each chromosome individually.
    """
    @wraps(func)
    @by_chrom
    @with_chrom
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def _interval_permute(df, chromosome_sizes):
    """
    Permute intervals not preserving size of gaps.
    """

    group_list = list()
    for chrom, group in df.groupby('chrom'):

        assert group.end.max() <= chromosome_sizes[chrom]

        segment_lengths = (group.end - group.start).tolist()
        total_gap = np.sum(group.start - group.end.shift())
        if np.isnan(total_gap): # in case there are no internal gaps (one segment)
            total_gap = 0
        else:
            total_gap = int(total_gap)
        if group.start.iloc[0] != 0:
            total_gap += group.start.iloc[0]
        if group.end.iloc[-1] != chromosome_sizes[chrom] + 1:
            total_gap += chromosome_sizes[chrom] + 1 - group.end.iloc[-1]

        assert total_gap >= len(segment_lengths)+1, (total_gap, len(segment_lengths)+1)
        idx = pd.Series(sorted(random.sample(range(total_gap), len(segment_lengths)+1)))
        gap_lengths = (idx - idx.shift()).dropna().tolist()

        random.shuffle(gap_lengths)
        random.shuffle(segment_lengths)
        borders = np.cumsum([j for i in zip(gap_lengths, segment_lengths) for j in i])
        starts, ends = borders[::2], borders[1::2]

        new_df = pd.DataFrame({'chrom': chrom, 'start': starts, 'end': ends})
        group_list.append(new_df)

    return pd.concat(group_list)


def bootstrap(chromosome_sizes: Union[str, dict], samples:int=100000, smaller:bool=False, return_boot:bool=False, cores:int=1):
    """
    Parameterized decorator that turns a function producing a statistic into one that also
    produces a p-value from bootstrapping. The bootstrapping resamples the
    intervals of the first argument for each chromosome independently. Only
    required argument to bootstrap is the name of the genome assembly used.

    Parameters
    ----------
    chromosome_sizes : 
        Name of a genome assembly or a dictionary mapping chromosomes to their lengths.
    samples : 
        Number of bootstrap samples to use.
    smaller :
        Whether to test for significantly small values of the statistic rather than large ones.
    return_boot :
        Whether to return the bootstrap samples too.
    cores :
        Number of CPU cores to use for computation, by default 1.
        
    Returns
    -------
    : 
        The decorated function returns a statistic and a p-value. A decorated function that takes data 
        frames with chrom, start, end columns and executes on each chromosome individually. 
    """

    if type(chromosome_sizes) is str:
        chromosome_sizes = chrom_sizes[chromosome_sizes]

    def decorator(func):
        @wraps(func)
        def wrapper(query, annot, **kwargs):

            stat = func(query, annot, **kwargs)

            try:
                from multiprocess import Pool
                multi = True
            except ImportError:
                multi = True

            if cores > 1 and multi:
                def _fun(query, annot, kwargs):
                    perm = _interval_permute(query, chromosome_sizes)
                    return func(perm, annot, **kwargs)
                with Pool(cores) as pool:
                    gen = pool.starmap(_fun, ((query, annot, kwargs) for _ in range(samples)))
                boot = list(gen)
            else:
                if cores > 1:
                    print("multiprocess library is required for multiprocessing:",
                          "    conda install conda-forge::multiprocess",
                          file=sys.stderr)
                boot = list()
                for i in range(samples):
                    perm = _interval_permute(query, chromosome_sizes)
                    boot.append(func(perm, annot, **kwargs))
            # boot = list()
            # for i in range(samples):
            #     perm = _interval_permute(query, chromosome_sizes)
            #     boot.append(func(perm, annot, **kwargs))
                    
            boot.sort()
            if smaller:
                p_value = bisect.bisect_left(boot, stat) / len(boot)
            else:
                p_value = (len(boot) - bisect.bisect_left(boot, stat)) / len(boot)
            if p_value == 0:
                sys.stderr.write('p-value is zero smaller than {}. Increase nr samples to get actual p-value.\n'.format(1/samples))

            if return_boot:
                return stat, p_value, boot
            else:
                return stat, p_value

        return wrapper
    return decorator
