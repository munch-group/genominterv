import numpy as np

from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

import bisect

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

def remap(query: Tuple[int], annot: List[tuple], relative=False, include_prox_coord=False, overlap_as_zero=False, span_as_zero=False) -> List[tuple]:
    """
    Remap the coordinates of a single interval in `query` to the distance from
    the closet interval in `annot`. Returns empty set if annot is empty for
    the chromosome. Intervals from `query` that overlap intervals in `annot`
    are discarded.

    Parameters
    ----------
    query : 
        A tuple of (start, end) coordinates.

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
