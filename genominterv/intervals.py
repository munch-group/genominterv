import pandas as pd
import numpy as np
import pandas
import matplotlib.pyplot as plt
from collections.abc import Callable
from typing import Any, TypeVar, List, Tuple, Dict, Union

from .interval_set_op import union, intersect, diff, collapse
from .decorators import genomic


def _plot_intervals(query=None, annot=None, **kwargs):

    import matplotlib.pyplot as plt

    tups = []
    if query is not None:
        tups.append(('query', query))
    if annot is not None:
        tups.append(('annot', annot))
    tups.extend(kwargs.items())
    tups = reversed(tups)

    df_list = []
    labels = []
    for label, df in tups:
        labels.append(label)
        df['label'] = label
        df_list.append(df)
    bigdf = pandas.concat(df_list)

    bigdf['chrom'] = pandas.Categorical(bigdf['chrom'], bigdf['chrom'].unique())
    bigdf['label'] = pandas.Categorical(bigdf['label'], bigdf['label'].unique())

    gr = bigdf.groupby('chrom', observed=False)

    fig, axes = plt.subplots(gr.ngroups, 1, figsize=(8, 1.5*gr.ngroups), 
                            sharey=True
                            #  sharex=True
                             )
    if type(axes) is not np.ndarray:
        # in case there is only one axis so it not returned as a list
        axes = [axes]
    
    # with plt.style.context(('default')):

    for i, chrom in enumerate(gr.groups):
        _df = gr.get_group(chrom)
        _gr = _df.groupby('label', observed=False)
        for y, label in enumerate(_gr.groups):
            try:
                df = _gr.get_group(label)
            except KeyError:
                continue
            y = np.repeat(y, df.index.size)
            axes[i].hlines(y, df.start.tolist(), df.end.tolist(), alpha=0.5, lw=2, colors=f'C{y[0]}')
            delta = len(labels)/10
            axes[i].vlines(df.start.tolist(), y-delta, y+delta, alpha=0.5, lw=2, colors=f'C{y[0]}')
            axes[i].vlines(df.end.tolist(), y-delta, y+delta, alpha=0.5, lw=2, colors=f'C{y[0]}')

        axes[i].spines['top'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        axes[i].set_yticks(list(range(len(labels))), labels)
        axes[i].tick_params(axis='y', which='both', left=False)
        axes[i].set_ylim(-1, len(labels)-0.7)
        # axes[i].set_xlim(df.start.min()-delta, df.end.max()+delta)
        if i != gr.ngroups-1:
            axes[i].tick_params(axis='x', which='both', bottom=False)

        axes[i].set_title(chrom, loc='left', fontsize=10)
    plt.tight_layout()


@genomic
def interval_diff(query: pandas.DataFrame, annot: pandas.DataFrame) -> pandas.DataFrame:
    """
    This function computes the difference between two sets of genomic intervals.
    The genomic intervals in each set must be non-overlapping. This can be
    achieved using :any:interval_collapse.

    Parameters
    ----------
    query : 
        Data frame with query intervals.
    annot : 
        Data frame with annotation intervals.        

    Returns
    -------
    :
        A data frame with chr, start, end columns representing the difference.


    See Also
    --------
    [](`genominterv.intervals.interval_union`), [](`genominterv.intervals.interval_intersect`), [](`genominterv.intervals.interval_collapse`).
    """

    return diff(query, annot)


@genomic
def interval_union(query: pandas.DataFrame, annot: pandas.DataFrame) -> pandas.DataFrame:
    """
    This function computes the union of two sets of genomic intervals. The genomic intervals
    in each set must be non-overlapping. This can be achieved using interval_collapse.

    Parameters
    ----------
    query : 
        Data frame with query intervals.
    annot : 
        Data frame with annotation intervals.        

    Returns
    -------
    :
        A data frame with chr, start, end columns representing the union.

    See Also
    --------
    [](`genominterv.intervals.interval_diff`), [](`genominterv.intervals.interval_intersect`), [](`genominterv.intervals.interval_collapse`).
    """

    return union(query, annot)


@genomic
def interval_intersect(query: pandas.DataFrame, annot: pandas.DataFrame) -> pandas.DataFrame:
    """
    This function computes the intersection of two sets of genomic intervals. The genomic intervals
    in each set must be non-overlapping. This can be achieved using interval_collapse.

    Parameters
    ----------
    query : 
        Data frame with query intervals.
    annot : 
        Data frame with annotation intervals.        

    Returns
    -------
    :
        A data frame with chr, start, end columns representing the intersection.

    See Also
    --------
    [](`genominterv.intervals.interval_diff`), [](`genominterv.intervals.interval_union`), [](`genominterv.intervals.interval_collapse`).
    """

    return intersect(query, annot)


@genomic
def interval_collapse(interv: pandas.DataFrame) -> pandas.DataFrame:
    """
    This function computes the union of intervals in a single set.

    Parameters
    ----------
    interv : 
        Data frame with intervals.

    Returns
    -------
    :
        A data frame with chr, start, end columns representing the union.

    See Also
    --------
    [](`genominterv.intervals.interval_diff`), [](`genominterv.intervals.interval_union`), [](`genominterv.intervals.interval_intersect`).
    """

    return collapse(interv)

# def ovl_interval_data(query, annot):

#     query_grouped = query.groupby('chrom')
#     annot_grouped = annot.groupby('chrom')

#     query_df_list = list()
#     annot_df_list = list()

#     for chrom, query_group in query_grouped:
#         annot_group = annot_grouped.get_group(chrom)

#         starts = query_group.start.tolist()
#         ends = query_group.end.tolist()

#         idx_list = list()    
#         annot_idx_list = list()
#         for tup in annot_group.itertuples():

#             start_idx = bisect.bisect_right(starts, tup.start) - 1
#             end_idx = bisect.bisect_right(starts, tup.end) - 1

#             if start_idx > -1 and tup.start < ends[start_idx]:
#                 idx_list.append(start_idx)
#                 annot_idx_list.append(tup.Index)
#             elif start_idx > -1 and tup.end < ends[start_idx]:
#                 idx_list.append(end_idx)
#                 annot_idx_list.append(tup.Index)

#         query_df_list.append(query_group.iloc[idx_list])
#         annot_df_list.append(annot_group.loc[annot_idx_list, ['start', 'end']])

#     query_data_overlap = (pandas.concat(query_df_list)
#                             .reset_index(drop=True)
#                          )
#     annot_intervals = (pandas.concat(annot_df_list)
#                         .reset_index(drop=True)
#                         .rename(columns={'start': 'ovl_start', 'end': 'ovl_end'})
#                         )
#     return pandas.concat([query_data_overlap, annot_intervals], axis=1)




# if __name__ == "__main__":


    # # print(remap((300, 400), [(0, 100), (500, 700), (10000, 11000)], include_prox_coord=True))

    # # q = (300, 400)
    # # print(q)
    # # print(remap(q, [          (500, 700), (10000, 11000)]))
    # # print()
    # # q = (300, 400)
    # # print(q)
    # # print(remap(q, [(0, 100), (500, 700), (10000, 11000)]))

    # a = [(0, 100), (500, 700), (10000, 11000)]
    # f = [-float('inf')] + [x for y in a for x in y] + [float('inf')] 

    # q = (300, 600)
    # print(q, a, f)
    # print(remap(q, a))
    # print()
    # q = (300, 600)
    # print(q, a, f)
    # print(remap(q, a))


    # # print(remap((200, 220), [(0, 100), (500, 700), (10000, 11000)]))
    # # print(remap((400, 600), [(0, 100), (500, 700), (10000, 11000)]))
    # # print(remap((400, 600), [(0, 100), (500, 700), (10000, 11000)], overlap_as_zero=True))

    # # print(remap((200, 220), [(0, 100), (500, 700), (10000, 11000)], overlap_as_zero=True))
    # # print(remap((400, 600), [(0, 100), (500, 700), (10000, 11000)], overlap_as_zero=True, span_as_zero=True))
    # # print(remap((400, 800), [(0, 100), (500, 700), (10000, 11000)], include_prox_coord=True))

    # assert 0

    # query = pandas.DataFrame(dict(chrom='X', start=[3, 5], end=[15, 7], extra=['foo', 'bar']))
    # print(query)
    # annot = pandas.DataFrame(dict(chrom='X', start=[1, 20], end=[2, 25]))
    # print(annot)
    # print(remap_interval_data(query, annot))

    # assert 0

    # query = pandas.DataFrame(dict(chrom='X', start=[3, 5], end=[22, 7], extra=['foo', 'bar']))
    # print(query)
    # annot = pandas.DataFrame(dict(chrom='X', start=[1, 20], end=[2, 25]))
    # print(annot)

    # print(ovl_interval_data(query, annot))

    # assert 0

    # # annotation
    # tp = [('chr1', 1, 3), ('chr1', 4, 10), ('chr1', 25, 30), ('chr1', 20, 27), ('chr2', 1, 10), ('chr2', 1, 3)]
    # annot = pandas.DataFrame.from_records(tp, columns=['chrom', 'start', 'end'])
    # print("annot\n", annot)

    # # query
    # tp = [('chr1', 8, 22), ('chr8', 14, 15)]
    # query = pandas.DataFrame.from_records(tp, columns=['chrom', 'start', 'end'])
    # print("query\n", query)

    # annot_collapsed = interval_collapse(annot)
    # print("annot_collapsed\n", annot_collapsed)

    # non_ovl_query = interval_diff(query, annot_collapsed)
    # print("non_ovl_query\n", non_ovl_query)

    # distances = interval_distance(non_ovl_query, annot_collapsed)
    # print("distances\n", distances)

    # print("distance test\n", proximity_test(non_ovl_query, annot_collapsed))


    # sys.exit()
    # # ##################################################################

    # print('jaccard:')
    # annot = pandas.DataFrame({'chrom': 'chr1', 'start': range(0, 1000000, 1000), 'end': range(100, 1000100, 1000)})
    # query = pandas.DataFrame({'chrom': 'chr1', 'start': range(50, 1000050, 1000), 'end': range(150, 1000150, 1000)})

    # print(annot)
    # print(query)

    # # print(interval_jaccard(query, annot, samples=10, chromosome_sizes={'chr1': 1500000, 'chr2': 1500000}))


    # annot = pandas.DataFrame({'chrom': 'chr1', 'start': [500, 2000], 'end': [1000, 2500]})
    # query = pandas.concat([pandas.DataFrame({'chrom': 'chr1', 'start': [1100, 1800], 'end': [1200, 1900]}),
    #                        pandas.DataFrame({'chrom': 'chr2', 'start': [1100, 1800], 'end': [1200, 1900]})
    #                        ])

    # print(annot)
    # print(query)

    # chromosome_sizes={'chr1': 1500000, 'chr2': 1500000}
    # @bootstrap(chromosome_sizes, samples=10)
    # def my_stat(a, b):
    #     return jaccard_stat(a, b)

    # print(my_stat(query, annot))

    # @genomic
    # def interval_computation(a, b):
    #     # stupid function that always returns:
    #     return [(1, 1)]

    # print(interval_computation(query, annot))

    # @bootstrap('hg19', samples=10)
    # def statistic(a, b):
    #     return 42

    # print(statistic(query, annot))





    # def overlap_count(query, annot):
    #       center = annot.start + (annot.end-annot.start)/2
    #       b = np.equal(np.searchsorted(annot.start, center) - 1,
    #                       np.searchsorted(annot.end, center, side='left')) & \
    #             (query.chrom == annot.chrom)

    #       return b.sum()

    # @bootstrap(chromosome_sizes, samples=1000)
    # def my_stat(a, b):
    #     return overlap_count(a, b)

    # print(my_stat(query, annot))

    ####################################################

    # annot_collapsed = (annot.groupby('chrom')
    #                    .apply(interval_collapse)
    #                    .reset_index(drop=True)
    #                    )
    # print("annot_collapsed\n", annot_collapsed)

    # non_ovl_query = (DataFrameList(query, annot_collapsed)
    #                  .groupby('chrom')
    #                  .apply(interval_diff)
    #                  .reset_index(drop=True)
    #                  )
    # print("non_ovl_query\n", non_ovl_query)

    # distances = (DataFrameList(non_ovl_query, annot_collapsed)
    #        .groupby('chrom')
    #        .apply(interval_distance)
    #        .reset_index(drop=True)
    #        )
    # print("distances\n", distances)