import numpy as np
from itertools import chain
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union
import pandas
import bisect

from .decorators import genomic

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return i-1
    raise ValueError

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return i-1
    raise ValueError

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError

def remap(query: Tuple[int], annot: List[tuple], relative=False, include_prox_coord=False, 
          overlap_as_zero=False, span_as_zero=False) -> List[tuple]:
    """
    Remap the coordinates of a single interval in `query` to the distance from
    the closet interval in `annot`. Returns empty set if annot is empty for
    the chromosome. Intervals from `query` that overlap intervals in `annot`
    are discarded.

    Parameters
    ----------
    query : 
        Query interval. A tuple of (start, end) coordinates.
    annot : 
        Data frame with annotation intervals. A list of tuples with (start, end) coordinates.
    relative : 
        Return relative distance (0-1) instead of absolute distance, by default False.
    include_prox_coord : 
        Include coordinates of the closest annotation segment, by default False.
    overlap_as_zero : 
        Set distance to zero if one end of a query segment overlaps an annotation segment, by default False.
        This does not apply to query segments embedded in or spanning on or more annotation segments.
    span_as_zero : 
        Set distance to zero if a query segment spans a single annotation segment, by default False.        

    Returns
    -------
    :
        A list of tuples with (start, end) coordinates.
    """

    if not (query and annot):
        return []

    query_start, query_end = query

    flat_coords = [-float('inf')] + [x for tup in annot for x in tup] + [float('inf')]
    start_idx = find_le(flat_coords, query_start)
    end_idx = find_ge(flat_coords, query_end)

    def start_found(idx): return not idx % 2
    def end_found(idx): return idx % 2

    if start_found(start_idx) and end_found(end_idx):
        if start_idx + 1 == end_idx:
            # start and end between same two segments
            interval_start = flat_coords[start_idx]
            interval_end = flat_coords[end_idx]
            interval_mid = (interval_start + interval_end) / 2
        elif start_idx + 3 == end_idx and span_as_zero:
            return [(0, 0)]
        else:
            # start and end on each side of more than one segment
            return []
    elif start_found(start_idx) and start_idx + 2 == end_idx and overlap_as_zero:
        # end is inside segment
        interval_start = flat_coords[start_idx]
        interval_end = flat_coords[end_idx-1]
        interval_mid = (interval_start + interval_end) / 2
        query_end = interval_end
    elif end_found(end_idx) and start_idx + 2 == end_idx and overlap_as_zero:
        # start is inside segment
        interval_start = flat_coords[start_idx+1]
        interval_end = flat_coords[end_idx]
        interval_mid = (interval_start + interval_end) / 2
        query_start = interval_start
    else:
        # both inside different segments
        return []

    scale = 1 / (interval_end - interval_start) if relative else 1

    if interval_mid < query_start:
        if include_prox_coord:
            remapped = [((query_end - interval_end) * scale, (query_start - interval_end) * scale,
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[end_idx], 
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[end_idx+1])]
        else:
            remapped = [((query_end - interval_end) * scale, (query_start - interval_end) * scale)]
    elif interval_mid >= query_end:
        if include_prox_coord:
            remapped = [((query_start - interval_start) * scale, (query_end - interval_start) * scale,
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[start_idx-1], 
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[start_idx])]
        else:
            remapped = [((query_start - interval_start) * scale, (query_end - interval_start) * scale)]            
    else:
        if include_prox_coord:
            remapped = [((query_start - interval_start) * scale, (interval_mid - interval_start) * scale,
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[start_idx-1], 
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[start_idx]),
                        ((query_end - interval_end) * scale, (interval_mid - interval_end) * scale,
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[end_idx], 
                        abs(interval_mid) == float('inf') and np.nan or flat_coords[end_idx+1])]
        else:
            remapped = [((query_start - interval_start) * scale, (interval_mid - interval_start) * scale), 
                        ((query_end - interval_end) * scale, (interval_mid - interval_end) * scale)]

    # FIXME: The stuff below should be done before the include_prox_coord stuff is done...

    # # compute remapped distance relative to the interval length (so that is is max 0.5)
    # if relative:
    #     if interval_start is None or interval_end is None:
    #         remapped = [(np.nan, np.nan)]
    #     else:
    #         interval_size = float(interval_end - interval_start)
    #         remapped = [(s/interval_size, e/interval_size) for (s, e) in remapped]

    return remapped



@genomic
def interval_distance(query: pandas.DataFrame, annot: pandas.DataFrame, relative:bool=False,
                      overlap_as_zero:bool=False, span_as_zero:bool=False) -> pandas.DataFrame:
    """
    Computes the distance from each query interval to the closest interval in
    annot. If a query interval overlaps the midpoint between two annot intervals
    it is split into two intervals proximal to each annot interval.    Intervals
    from ``query`` that overlap intervals in ``annot`` are discarded.

    Parameters
    ----------
    query : 
        Data frame with query intervals.
    annot : 
        Data frame with annotation intervals.        
    relative : 
        Return relative distance (0-1) instead of absolute distance, by default False.
    overlap_as_zero : 
        Set distance to zero if one end of a query segment overlaps an annotation segment, by default False.
        This does not apply to query segments embedded in or spanning on or more annotation segments.
    span_as_zero : 
        Set distance to zero if a query segment spans a single annotation segment, by default False.   

    Returns
    -------
    :
        A data frame with remapped intervals.

    See Also
    --------
    If you want to retain the original columns in `query`, use [](`~genominterv.remapping.remap_interval_data`).
    """
    return list(chain.from_iterable(
        remap(q, annot, overlap_as_zero=overlap_as_zero, span_as_zero=span_as_zero, relative=relative) for q in query))


def remap_interval_data(query: pandas.DataFrame, annot: pandas.DataFrame, relative:bool=False,
                      overlap_as_zero:bool=False, span_as_zero:bool=False) -> pandas.DataFrame:
    """
    Computes the distance from each query interval to the closest interval
    in annot. Original coordinates are preserved as `orig_start` and
    `orig_end` columns. If a query interval overlaps the midpoint between two
    annot intervals it is split into two intervals proximal to each
    annot interval, thus contributing two rows to the returned data frame.
    Intervals from `query` that overlap intervals in `annot` are discarded.

    Parameters
    ----------
    query : 
        Data frame with query intervals.
    annot : 
        Data frame with annotation intervals.        
    relative : 
        Return relative distance (0-1) instead of absolute distance, by default False.
    overlap_as_zero : 
        Set distance to zero if one end of a query segment overlaps an annotation segment, by default False.
        This does not apply to query segments embedded in or spanning on or more annotation segments.
    span_as_zero : 
        Set distance to zero if a query segment spans a single annotation segment, by default False.   

    Returns
    -------
    :
        A data frame with remapped intervals.

    See Also
    --------
    If you do not want to retain the original columns in `query`, use [](`~genominterv.remapping.interval_distance`).
    """

    annot_grouped = annot.groupby('chrom')

    df_list = list()
    column_names = tuple(query.columns.values)
    for chrom, group in query.groupby('chrom'):

        chrom_annot = annot_grouped.get_group(chrom)
        annot_tups = [tuple(t) for t in chrom_annot[['start', 'end']].itertuples(index=False)]

        remapped = list()
        # for index, row in group.iterrows():            
            # start, end = (row['start'], row['end'])
        for row in group.itertuples(index=False):            
            start, end = row.start, row.end
            for remapped_start, remapped_end, start_prox, end_prox in remap(
                    (start, end), annot_tups, include_prox_coord=True,
                    overlap_as_zero=overlap_as_zero, span_as_zero=span_as_zero, relative=relative
                    ):
                remapped.append((remapped_start, remapped_end, start_prox, end_prox) + tuple(row))

        df = pandas.DataFrame().from_records(remapped, 
                columns=('start_remap', 'end_remap', 'start_prox', 'end_prox') + column_names)
            # df = pandas.DataFrame().from_records(remapped, columns=['idx', 'start_remap', 'end_remap']).set_index('idx')
            # df = pandas.merge(group, df, right_index=True, left_index=True)

        df_list.append(df)

    df = (pandas.concat(df_list)
            .reset_index(drop=True)
            .rename(columns={'start': 'start_orig',
                            'end': 'end_orig'})
            .rename(columns={'start_remap': 'start',
                            'end_remap': 'end'})
            )

    df['start_orig'] = df['start_orig'].astype('Int64')
    df['end_orig'] = df['end_orig'].astype('Int64')
    df['start_prox'] = df['start_prox'].astype('Int64')
    df['end_prox'] = df['end_prox'].astype('Int64')

    return df


