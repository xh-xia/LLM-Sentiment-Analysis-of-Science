import networkx as nx
import numpy as np
import pickle
import os
import itertools
from tqdm import tqdm
from collections import defaultdict

from external_methods import process_batch_outputs_fc, process_response_2
from helper_functions import cosine_sim, get_collab_distance, loadPKL, savePKL


def make_cite2sent_from_sentence_data(dir_in, dir_out):
    """

    Arg
    ---
    - dir_in (str): folder containing sentrow2edgeinfo
        key: row index in sentences sent to ChatGPT (i.e., sentences2rate-CGPT.txt)
        val: [idx, sent_idx2pmcid_lookup[idx], [s for s in c], None]

    Return & Save
    -------------
    - cite2sent (dict):
    key: (pmcid_i, pmcid_j); citation pair
    val: []
        each member of the list is row idx (for sentences2rate-CGPT.txt) of an instance of citation
        paper i can cite paper j multiple times
        once we get the sentiment, we will replace the row index with sentiment value
        each time we modify this dict we increment its suffix index by 1
        so this function saves cite2sent as "cite2sent_0.pkl"
    """
    with open(os.path.join(dir_in, "sentrow2edgeinfo.pkl"), "rb") as f:
        sentrow2edgeinfo = pickle.load(f)
    cite2sent = dict()
    for row_idx, val_list in sentrow2edgeinfo.items():
        pmcid_i = val_list[1]
        for pmcid_j in val_list[2]:
            e = (pmcid_i, pmcid_j)
            if e not in cite2sent:
                cite2sent[e] = [row_idx]
            else:
                cite2sent[e].append(row_idx)

    with open(os.path.join(dir_out, "cite2sent_0.pkl"), "wb") as f:
        pickle.dump(cite2sent, f)
    return cite2sent


def update_cite2sent_from_row2rate(dir_TEMP, dir_batch):
    """
    Args
    ----
    - dir_TEMP (str); folder path containing cite2sent_0 (dict):
        Update empirical sentiment in cite2sent_0
        using row2rate_reason (in dir_batch) and sentrow2edgeinfo (in dir_TEMP),
        former of which contains values that are raw output from ChatGPT
        latter of which is a bookkeeping file.
        row2rate_reason: val is len-2 list, [0] is processed rating, [1] is rationale.
    Save
    ----
    - cite2sent (dict):
        key: (pmcid_i, pmcid_j); citation edge
        val: []
            each member of the list is empirical sentiment of an instance of citation
            paper i can cite paper j multiple times
            this function saves cite2sent as "cite2sent_1.pkl" in dir_TEMP
    """
    row2rate_reason = loadPKL(dir_batch, "row2rate_reason_processed")
    with open(os.path.join(dir_TEMP, "sentrow2edgeinfo.pkl"), "rb") as f:
        sentrow2edgeinfo = pickle.load(f)
    with open(os.path.join(dir_TEMP, "cite2sent_0.pkl"), "rb") as f:
        cite2sent = pickle.load(f)
    if len(row2rate_reason) != len(sentrow2edgeinfo):
        raise Exception(f"<row2rate_reason> has {len(row2rate_reason)} lines, but <sentrow2edgeinfo> has {len(sentrow2edgeinfo)} lines")
    for i in sentrow2edgeinfo.keys():
        citing_idx = int(sentrow2edgeinfo[i][1])
        cited_idxs = sentrow2edgeinfo[i][2]
        rate = row2rate_reason[i][0]
        for ci in cited_idxs:  # there could be multiple cited articles
            cited_idx = int(ci)
            j = cite2sent[(citing_idx, cited_idx)].index(i)
            if rate > 0:
                cite2sent[(citing_idx, cited_idx)][j] = "POS"
            elif rate < 0:
                cite2sent[(citing_idx, cited_idx)][j] = "NEG"
            else:
                cite2sent[(citing_idx, cited_idx)][j] = "NEU"

    # Convert string sentiment to integer sentiment.
    for k in list(cite2sent.keys()):
        for s in range(len(cite2sent[k])):
            str_ = cite2sent[k][s]
            if str_ == "POS":
                cite2sent[k][s] = 1
            elif str_ == "NEU":
                cite2sent[k][s] = 0
            elif str_ == "NEG":
                cite2sent[k][s] = -1
            else:
                raise Exception(f"<cite2sent> has a citation at row_idx={str_} that is not populated by empirical sentiment")

    with open(os.path.join(dir_TEMP, "cite2sent_1.pkl"), "wb") as f:
        pickle.dump(cite2sent, f)


def update_cite2sent_hierarchy_rule(dir_TEMP, dir_dict):
    """
    Args
    ----
    - dir_TEMP (str); folder path containing cite2sent_1 (dict):
        Update empirical sentiment in cite2sent_1 by converting multiple sentiments into one:
            if a citation pair has one NEG citation, its value is -1
            elif a citation pair has one POS citation, its value is 1
            else, its value is 0
    Save
    ----
    - cite2sent (dict): save to dir_dict
        key: (pmcid_i, pmcid_j); citation edge
        val: int
            Paper i can cite paper j multiple times, but we apply above hierarchy rule,
            so now it only has exactly one sentiment.
            This function saves cite2sent as "cite2sent_2.pkl" in dir_dict.
            This dict now holds the "empirical sentiment data".
    - cite2ns (dict): save to dir_TEMP
        key: (pmcidi, pmcidj)
        val (int): num of citation sentences from article i to article j.
    """
    with open(os.path.join(dir_TEMP, "cite2sent_1.pkl"), "rb") as f:
        cite2sent = pickle.load(f)
    cite2ns = {k: len(v) for k, v in cite2sent.items()}
    for k in list(cite2sent.keys()):
        if -1 in cite2sent[k]:
            cite2sent[k] = -1
        elif 1 in cite2sent[k]:
            cite2sent[k] = 1
        else:
            cite2sent[k] = 0

    with open(os.path.join(dir_dict, "cite2sent_2.pkl"), "wb") as f1, open(os.path.join(dir_TEMP, "cite2ns.pkl"), "wb") as f2:
        pickle.dump(cite2sent, f1)
        pickle.dump(cite2ns, f2)


def make_paper2meta(cite_pairs, dir_article_meta, dir_out):
    """cite_pairs is list-like of citation pair: (pmcid_i, pmcid_j)"""
    papers = set()
    for e in cite_pairs:
        papers.add(e[0])
        papers.add(e[1])
    papers = sorted(list(papers))
    with open(os.path.join(dir_article_meta, "article_meta.pkl"), "rb") as f:
        article_meta = pickle.load(f)
    paper2meta = {p: article_meta[p] for p in papers}
    with open(os.path.join(dir_out, "paper2meta.pkl"), "wb") as f:
        pickle.dump(paper2meta, f)
    return paper2meta


def save_cite2title_sim(dir_dict, cite_pairs, dir_out):
    """cite_pairs is list-like of citation pair: (pmcid_i, pmcid_j)"""
    with open(os.path.join(dir_dict, "paper2embed.pkl"), "rb") as f:
        paper2embed = pickle.load(f)
    cite2title_sim = {e: cosine_sim(paper2embed[e[0]], paper2embed[e[1]]) for e in cite_pairs}
    with open(os.path.join(dir_out, "cite2title_sim.pkl"), "wb") as f:
        pickle.dump(cite2title_sim, f)


def save_cite2sent_null_param(dir_dict, dir_temp, maxN=15, n_min_samp=500):
    """
    Make null model.
    Group citation pairs by their number of citations, title similarity, and paper type.
    Within each group, we tally the number of three types of sentiment
    and estimate a categorical distribution using maximum likelihood estimation.
    The estimated parameters (list of 3 values, one for each sentiment) are val in the dict.
        Title similarity quantile is found using title similarity for all citation pairs.
        Only estimate upto 15 num of sentences, for those beyond, use ns=15
        Only estimate for bins whose n>=n_min_samp.
        Else, using estimate for marginal bins (paper type x num of cite) whose n>=n_min_samp.

    Kwargs
    ------
    - maxN (int):
        Maximum number of sentences considered in the null model
    - n_min_samp (int):
        Minimal number of samples in each bin. Default to 500.
    Save
    ----
    - cite2sent_null_param (dict):
        key: Same as cite2sent, (pmcid_i, pmcid_j), citation pair
        val: [pr(POS), pr(NEU), pr(NEG)]
    """
    sent2idx = {1: 0, 0: 1, -1: 2}
    cite2sent_2 = loadPKL(dir_dict, "cite2sent_2")
    cite2ns = loadPKL(dir_temp, "cite2ns")
    cite2title_sim = loadPKL(dir_temp, "cite2title_sim")
    paper2meta = loadPKL(dir_dict, "paper2meta")

    # Find 3 ingredients used to find groups.
    pt_arr = []  # Paper type: 0 for research, 1 for review
    ns_arr = []  # Num of sentences: 1,2,...,15
    ts_arr = []  # Title similarity
    st_arr = []  # Empirical sentiment: 1,0,-1; one for each citation pair.
    cite2grp = dict()  # key: citation pair; val: [pt, ns, ts]
    cites = []  # Citation pairs, keeping track of the order.
    for e, sentiment in cite2sent_2.items():
        if paper2meta[e[0]]["article-type"] == "research-article":
            pt_arr.append(0)
        elif paper2meta[e[0]]["article-type"] == "review-article":
            pt_arr.append(1)
        else:
            raise Exception(f"Unknown article type: {paper2meta[e[0]]['article-type']}.")
        ns_arr.append(cite2ns[e])
        ts_arr.append(cite2title_sim[e])
        st_arr.append(sentiment)
        cite2grp[e] = [pt_arr[-1], ns_arr[-1], ts_arr[-1]]
        cites.append(e)

    pt_arr = np.array(pt_arr)
    ns_arr = np.array(ns_arr)
    ts_arr = np.array(ts_arr)
    ts0, ts1 = np.quantile(ts_arr, [1 / 3, 2 / 3])
    # Turn title similarity into quantile groups from low to high: 0,1,2.
    ts_arr = np.where(ts_arr <= ts0, 0, np.where(ts_arr > ts1, 2, 1))
    for i, e in enumerate(cites):
        cite2grp[e][-1] = ts_arr[i]

    # Estimate categorical distribution for each group.
    pt_types = np.unique(pt_arr)
    ns_types = np.arange(1, maxN + 1)  # Only first maxN ns.
    ts_types = np.unique(ts_arr)
    # dim0: sentiment; dim1: paper type; dim2: num of sentences; dim3: title sim quantile grp.
    counts = np.zeros((3, len(pt_types), len(ns_types), len(ts_types)))
    for idx, sent in enumerate(st_arr):
        if ns_arr[idx] > maxN:  # Skip citation pairs that have ns > maxN during estimation.
            continue
        counts[sent2idx[sent], pt_arr[idx], ns_arr[idx] - 1, ts_arr[idx]] += 1

    counts_3_sent = np.sum(counts, axis=0, keepdims=True)
    n_less_list = []
    categorical = counts / counts_3_sent
    for pt in pt_types:
        for ns in ns_types:
            for ts in ts_types:
                n_less = counts_3_sent[0, pt, ns - 1, ts]
                if n_less < n_min_samp:
                    # Find marginal over ts.
                    new_count = np.sum(counts[:, pt, ns - 1, :], axis=-1)
                    new_ratio = new_count / np.sum(new_count)
                    # Replace the estimate with marginal.
                    categorical[:, pt, ns - 1, ts] = new_ratio
                    # Tally.
                    n_less_list.append(n_less)

    print(f"In bins whose sample size < {n_min_samp}")
    print(f"there are {len(n_less_list)} bins, {np.sum(n_less_list)} citation pairs,")
    print(f"taking up {np.sum(n_less_list)/np.sum(counts_3_sent)*100:.2f}% of all pairs.")
    # Populate estimated sentiment probability for each citation pair given its group.
    cite2sent_null_param = {p: None for p in cite2grp}
    for p, (pt, ns, ts) in cite2grp.items():
        if ns > maxN:
            cite2sent_null_param[p] = categorical[:, pt, maxN - 1, ts]
        else:
            cite2sent_null_param[p] = categorical[:, pt, ns - 1, ts]

    savePKL(dir_dict, f"cite2sent_null_param{f'_ns={maxN}' if maxN != 15 else ''}", cite2sent_null_param)


def save_cite2sent_null_param_only(dir_dict, dir_temp, maxN=15, n_min_samp=500, which_only=None):
    """
    Make null model, except we remove review/research papers.
    Group citation pairs by their number of citations, title similarity, and paper type.
    Within each group, we tally the number of three types of sentiment
    and estimate a categorical distribution using maximum likelihood estimation.
    The estimated parameters (list of 3 values, one for each sentiment) are val in the dict.
        Title similarity quantile is found using title similarity for all citation pairs.
        Only estimate upto 15 num of sentences, for those beyond, use ns=15
        Only estimate for bins whose n>=n_min_samp.
        Else, using estimate for marginal bins (paper type x num of cite) whose n>=n_min_samp.

    Kwargs
    ------
    - maxN (int):
        Maximum number of sentences considered in the null model
    - n_min_samp (int):
        Minimal number of samples in each bin. Default to 500.
    - which_only (str):
        which paper type to make null model on exclusively.
    Save
    ----
    - cite2sent_null_param (dict):
        key: Same as cite2sent, (pmcid_i, pmcid_j), citation pair
        val: [pr(POS), pr(NEU), pr(NEG)]
    """
    sent2idx = {1: 0, 0: 1, -1: 2}
    cite2sent_2 = loadPKL(dir_dict, "cite2sent_2")
    cite2ns = loadPKL(dir_temp, "cite2ns")
    cite2title_sim = loadPKL(dir_temp, "cite2title_sim")
    paper2meta = loadPKL(dir_dict, "paper2meta")
    
    if which_only is None:
        which_only = "research-article"

    # Find 2 ingredients used to find groups.
    ns_arr = []  # Num of sentences: 1,2,...,15
    ts_arr = []  # Title similarity
    st_arr = []  # Empirical sentiment: 1,0,-1; one for each citation pair.
    cite2grp = dict()  # key: citation pair; val: [ns, ts]
    cites = []  # Citation pairs, keeping track of the order.
    for e, sentiment in cite2sent_2.items():
        if paper2meta[e[0]]["article-type"] != which_only:
            continue  # Only keep research papers (citing papers only, same as the normal ver.).
        ns_arr.append(cite2ns[e])
        ts_arr.append(cite2title_sim[e])
        st_arr.append(sentiment)
        cite2grp[e] = [ns_arr[-1], ts_arr[-1]]
        cites.append(e)

    ns_arr = np.array(ns_arr)
    ts_arr = np.array(ts_arr)
    ts0, ts1 = np.quantile(ts_arr, [1 / 3, 2 / 3])
    # Turn title similarity into quantile groups from low to high: 0,1,2.
    ts_arr = np.where(ts_arr <= ts0, 0, np.where(ts_arr > ts1, 2, 1))
    for i, e in enumerate(cites):
        cite2grp[e][-1] = ts_arr[i]

    # Estimate categorical distribution for each group.
    ns_types = np.arange(1, maxN + 1)  # Only first maxN ns.
    ts_types = np.unique(ts_arr)
    # dim0: sentiment; dim1: num of sentences; dim2: title sim quantile grp.
    counts = np.zeros((3, len(ns_types), len(ts_types)))
    for idx, sent in enumerate(st_arr):
        if ns_arr[idx] > maxN:  # Skip citation pairs that have ns > maxN during estimation.
            continue
        counts[sent2idx[sent], ns_arr[idx] - 1, ts_arr[idx]] += 1

    counts_3_sent = np.sum(counts, axis=0, keepdims=True)
    n_less_list = []
    categorical = counts / counts_3_sent
    for ns in ns_types:
        for ts in ts_types:
            n_less = counts_3_sent[0, ns - 1, ts]
            if n_less < n_min_samp:
                # Find marginal over ts.
                new_count = np.sum(counts[:, ns - 1, :], axis=-1)
                new_ratio = new_count / np.sum(new_count)
                # Replace the estimate with marginal.
                categorical[:, ns - 1, ts] = new_ratio
                # Tally.
                n_less_list.append(n_less)

    print(f"In bins whose sample size < {n_min_samp}")
    print(f"there are {len(n_less_list)} bins, {np.sum(n_less_list)} citation pairs,")
    print(f"taking up {np.sum(n_less_list)/np.sum(counts_3_sent)*100:.2f}% of all pairs.")
    # Populate estimated sentiment probability for each citation pair given its group.
    cite2sent_null_param = {p: None for p in cite2grp}
    for p, (ns, ts) in cite2grp.items():
        if ns > maxN:
            cite2sent_null_param[p] = categorical[:, maxN - 1, ts]
        else:
            cite2sent_null_param[p] = categorical[:, ns - 1, ts]

    savePKL(dir_dict, f"cite2sent_null_param_{which_only}", cite2sent_null_param)


def save_cite2sent_null_param_pre_agg(dir_dict, dir_temp, maxN=15, n_min_samp=500):
    """
    Make null model. Difference from the vanilla function is that:
    This one tallies number of sentiment at pre-aggregation level, because this is for citation level sentiment ratio.
    There's some differences of pre-aggregation sentiment in num of sentences, so 3 factors kept.
    A nice thing about using pre-agg is that sample size is a lot larger, since there's multiple instances per pair, as opposed to always 1 in the vanilla function.
    So only cite2sent_, st_arr, and counts are changed, the rest is identical.
    a) vanilla: sentiment on a pair level (post-agg)
    b) pre-agg: sentiment on individual citation sentence of a pair (pre-agg)
    both follow a 3-category distribution, estimated from larger grouping.
    Group citation pairs by their number of citations, title similarity, and paper type.
    Within each group, we tally the number of three types of sentiment at pre-aggregation level.
    and estimate a categorical distribution using maximum likelihood estimation.
    The estimated parameters (list of 3 values, one for each sentiment) are val in the dict.
        Title similarity quantile is found using title similarity for all citation pairs.
        Only estimate upto 15 num of sentences, for those beyond, use ns=15
        Only estimate for bins whose n>=n_min_samp.
        Else, using estimate for marginal bins (paper type x num of cite) whose n>=n_min_samp.

    Kwargs
    ------
    - maxN (int):
        Maximum number of sentences considered in the null model
    - n_min_samp (int):
        Minimal number of samples in each bin. Default to 500.
    Save
    ----
    - cite2sent_null_param (dict):
        key: Same as cite2sent, (pmcid_i, pmcid_j), citation pair
        val: [pr(POS), pr(NEU), pr(NEG)]
    """
    sent2idx = {1: 0, 0: 1, -1: 2}
    cite2sent_1 = loadPKL(dir_temp, "cite2sent_1")  # Note the changes from sent_2 (post-agg) to sent_1 (pre-agg).
    cite2ns = loadPKL(dir_temp, "cite2ns")
    cite2title_sim = loadPKL(dir_temp, "cite2title_sim")
    paper2meta = loadPKL(dir_dict, "paper2meta")

    # Find 3 ingredients used to find groups.
    pt_arr = []  # Paper type: 0 for research, 1 for review
    ns_arr = []  # Num of sentences: 1,2,...,15
    ts_arr = []  # Title similarity
    st_arr = []  # Empirical sentiment: nested len-3 list of counts for 1,0,-1; one list for each citation pair.
    cite2grp = dict()  # key: citation pair; val: [pt, ns, ts]
    cites = []  # Citation pairs, keeping track of the order.
    for e, sentiments in cite2sent_1.items():
        if paper2meta[e[0]]["article-type"] == "research-article":
            pt_arr.append(0)
        elif paper2meta[e[0]]["article-type"] == "review-article":
            pt_arr.append(1)
        else:
            raise Exception(f"Unknown article type: {paper2meta[e[0]]['article-type']}.")
        ns_arr.append(cite2ns[e])
        ts_arr.append(cite2title_sim[e])
        l_st = [0, 0, 0]  # Count of sentiment for 1, 0, -1, in this order.
        for s in sentiments:
            l_st[sent2idx[s]] += 1
        st_arr.append(l_st)
        cite2grp[e] = [pt_arr[-1], ns_arr[-1], ts_arr[-1]]
        cites.append(e)

    pt_arr = np.array(pt_arr)
    ns_arr = np.array(ns_arr)
    ts_arr = np.array(ts_arr)
    ts0, ts1 = np.quantile(ts_arr, [1 / 3, 2 / 3])
    # Turn title similarity into quantile groups from low to high: 0,1,2.
    ts_arr = np.where(ts_arr <= ts0, 0, np.where(ts_arr > ts1, 2, 1))
    for i, e in enumerate(cites):
        cite2grp[e][-1] = ts_arr[i]

    # Estimate categorical distribution for each group.
    pt_types = np.unique(pt_arr)
    ns_types = np.arange(1, maxN + 1)  # Only first maxN ns.
    ts_types = np.unique(ts_arr)
    # dim0: sentiment; dim1: paper type; dim2: num of sentences; dim3: title sim quantile grp.
    counts = np.zeros((3, len(pt_types), len(ns_types), len(ts_types)))
    for idx, sent in enumerate(st_arr):
        if ns_arr[idx] > maxN:  # Skip citation pairs that have ns > maxN during estimation.
            continue
        for s in [1, 0, -1]:
            counts[sent2idx[s], pt_arr[idx], ns_arr[idx] - 1, ts_arr[idx]] += sent[sent2idx[s]]

    counts_3_sent = np.sum(counts, axis=0, keepdims=True)
    n_less_list = []
    categorical = counts / counts_3_sent
    for pt in pt_types:
        for ns in ns_types:
            for ts in ts_types:
                n_less = counts_3_sent[0, pt, ns - 1, ts]
                if n_less < n_min_samp:
                    # Find marginal over ts.
                    new_count = np.sum(counts[:, pt, ns - 1, :], axis=-1)
                    new_ratio = new_count / np.sum(new_count)
                    # Replace the estimate with marginal.
                    categorical[:, pt, ns - 1, ts] = new_ratio
                    # Tally.
                    n_less_list.append(n_less)

    print(f"In bins whose sample size < {n_min_samp}")
    print(f"there are {len(n_less_list)} bins, {np.sum(n_less_list)} citation pairs,")
    print(f"taking up {np.sum(n_less_list)/np.sum(counts_3_sent)*100:.2f}% of all pairs.")
    # Populate estimated sentiment probability for each citation pair given its group.
    cite2sent_null_param = {p: None for p in cite2grp}
    for p, (pt, ns, ts) in cite2grp.items():
        if ns > maxN:
            cite2sent_null_param[p] = categorical[:, pt, maxN - 1, ts]
        else:
            cite2sent_null_param[p] = categorical[:, pt, ns - 1, ts]

    savePKL(dir_dict, "cite2sent_null_param_pre_agg", cite2sent_null_param)


def save_g_coau_t2(dir_dict, weight_nocollab=None, weight_type=None, no_mid_mid=False, lag=0):
    """build a time-varying/dependent coauthorship network (undirected)
    If collaborated, then there's an edge.
    Edge attr is collaboration year y.
    It's cumulative, meaning at each year, it concerns all collaborated papers thus far, excluding this year (if lag=1)
    e.g., collab network at year 2024 is network of collaborations upto and including 2023, but not 2024

    - weight_nocollab (number):
        default weight when there's no collaboration (yet). Default to np.inf.
    - weight_type (str): what to use for edge weights
        - "binary": simpliest, if 2 authors have shown up on at least 1 paper, w=1; also the one used in sentiment project.
        - "cumulative": each time 2 authors show up on a paper, w+=1.
    - no_mid_mid (bool): If True, excluded middle-middle author edges, meaning only F-L, F-M, L-M, but no M-M.
    - lag (int): 1 means year x collab network uses data upto and including x-1, excluding x.
    """
    if weight_nocollab is None:
        weight_nocollab = np.inf
    if weight_type is None:
        weight_type = "binary"
    if no_mid_mid:
        paper2first_author = loadPKL(dir_dict, "paper2first_author")
        paper2last_author = loadPKL(dir_dict, "paper2last_author")

        def _nonce_is_mid_mid(aui, auj, fau, lau):
            return (aui != fau) and (aui != lau) and (auj != fau) and (auj != lau)

    weight_type = weight_type.casefold()
    assert weight_type in {"binary", "cumulative"}, f"unrecognized weight_type: {weight_type}"
    try:
        paper2year = loadPKL(dir_dict, "paper2year")
    except:
        paper2year = loadPKL(dir_dict, "paper2journal_and_year")
        paper2year = {k: v[1] for k, v in paper2year.items()}
    paper2author_s = loadPKL(dir_dict, "paper2author_s")
    paper2year = {k: v for k, v in paper2year.items() if k in paper2author_s}  # Only papers in paper2author_s.
    years = []
    for p, year in paper2year.items():
        years.append(year)
    year_min = min(years)
    year_max = max(years)
    year_lim = (year_min, year_max)
    # print(dict(sorted(Counter(years).items(), key=lambda x: x[0])))

    g_coau_t = nx.Graph()  # undirected
    # Every author (not just last authors) shows up in the coauthorship network.
    g_coau_t.add_nodes_from(set([au for aus in paper2author_s.values() for au in aus]))

    # Non-cumulative first.
    for y in tqdm(range(year_min, year_max + 1)):
        for p, p_year in paper2year.items():
            if p_year != y:
                continue
            for aui, auj in itertools.combinations(paper2author_s[p], 2):
                if no_mid_mid and _nonce_is_mid_mid(aui, auj, paper2last_author[p], paper2first_author[p]):
                    continue  # Skip middle-middle author edge.
                try:
                    g_coau_t.edges[aui, auj]
                except KeyError:
                    g_coau_t.add_edge(aui, auj)
                    g_coau_t.edges[aui, auj][y] = 1
                    continue
                try:
                    g_coau_t.edges[aui, auj][y] += 1
                except KeyError:
                    g_coau_t.edges[aui, auj][y] = 1

    # Then we go from most recent year to the oldest as we turn it into cumulative.
    # Because we loop through all collaboration edges, there will always be at least one instance of collaboration,
    # as in year_first won't remain np.inf.
    # year_first is the earliest year aui,auj collab'ed.
    if weight_type == "binary":
        for aui, auj, d in tqdm(g_coau_t.edges(data=True)):
            # years_collab = np.zeros(year_max - year_min + 1, dtype=int)
            year_first = np.inf  # First/Earliest year that they have collaborated.
            for y, c in d.items():  # c is collaboration count at year y.
                # years_collab[y - year_min] = c
                if year_first > y:
                    year_first = y
            year_first = int(year_first)
            for y in range(year_max + lag, year_first - 1 + lag, -1):
                # shortest path is distance, collab times is inverse, so this is commented out, so is years_collab
                # g_coau_t.edges[aui, auj][y] = np.sum(years_collab[0 : y - year_min])
                g_coau_t.edges[aui, auj][y] = 1
            for y in range(year_first - 1 + lag, year_min - 1 + lag, -1):
                g_coau_t.edges[aui, auj][y] = weight_nocollab  # Haven't collaborated yet.
    elif weight_type == "cumulative":
        for aui, auj, d in tqdm(g_coau_t.edges(data=True)):
            years_collab = np.zeros(year_max - year_min + 1, dtype=int)
            year_first = np.inf  # First/Earliest year that they have collaborated.
            for y, c in d.items():  # c is collaboration count at year y.
                years_collab[y - year_min] = c
                if year_first > y:
                    year_first = y
            year_first = int(year_first)
            for y in range(year_max + lag, year_first - 1 + lag, -1):
                g_coau_t.edges[aui, auj][y] = np.sum(years_collab[0 : y - year_min])
            for y in range(year_first - 1 + lag, year_min - 1 + lag, -1):
                g_coau_t.edges[aui, auj][y] = weight_nocollab  # Haven't collaborated yet.

    # savePKL(dir_dict, f"g_coau_t-{weight_type}-{'no' if no_mid_mid else 'has'}_mid_mid", g_coau_t)
    savePKL(dir_dict, "g_coau_t", g_coau_t)


def save_cite2distance(cite_pairs, dir_dict):
    """
    "distance" is collaboration distance
    which is shortest path length in collab network at the time of citation.
    cite_pairs is list-like of citation pair: (pmcid_i, pmcid_j)
    """
    g_coau_t = loadPKL(dir_dict, "g_coau_t")
    paper2last_author = loadPKL(dir_dict, "paper2last_author")
    paper2year = loadPKL(dir_dict, "paper2year")
    # Manually create a subgraph view of each year's collab network.
    # In the earliest year, no one will have collaborate in the previous year yet since we don't have the data.
    # In the latest year, the collab network is 1 year after, but we wouldn't use it since it's beyond the data.
    year_min = min([x for x in paper2year.values()])
    year_max = max([x for x in paper2year.values()])
    if year_min != 1998 or year_max != 2023:
        raise Exception(f"paper2year range != [1998,2023], but {(year_min, year_max)}, change viewxxxx and this if condition accordingly")
    view1998 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][1998] == 1)
    view1999 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][1999] == 1)
    view2000 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2000] == 1)
    view2001 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2001] == 1)
    view2002 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2002] == 1)
    view2003 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2003] == 1)
    view2004 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2004] == 1)
    view2005 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2005] == 1)
    view2006 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2006] == 1)
    view2007 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2007] == 1)
    view2008 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2008] == 1)
    view2009 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2009] == 1)
    view2010 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2010] == 1)
    view2011 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2011] == 1)
    view2012 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2012] == 1)
    view2013 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2013] == 1)
    view2014 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2014] == 1)
    view2015 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2015] == 1)
    view2016 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2016] == 1)
    view2017 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2017] == 1)
    view2018 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2018] == 1)
    view2019 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2019] == 1)
    view2020 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2020] == 1)
    view2021 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2021] == 1)
    view2022 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2022] == 1)
    view2023 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2023] == 1)

    cite2CD = {e: np.nan for e in cite_pairs}
    for e in tqdm(cite_pairs):
        aui = paper2last_author[e[0]]
        auj = paper2last_author[e[1]]

        # most recent year + 1
        # cite2CD[e] = get_collab_distance(locals()[f"view{year_max + 1}"], aui, auj, weight=None)
        # year of citation
        cite2CD[e] = get_collab_distance(locals()[f"view{paper2year[e[0]]}"], aui, auj, weight=None)

    savePKL(dir_dict, "cite2distance", cite2CD)


def get_ratio_mat_rel(sent_arr):
    """
    Calculate sentiment ratio percentage change relative to null model.
    Args:
        sent_arr (2D np.arr):
            row: sample (citation pair)
            col: emp and all null replica sentiments
    Return:
        ratio_mat (2D np.arr): shape (3, n_rep)
    """
    n_rep = sent_arr.shape[1] - 1  # Num of null replica.
    # Row: sentiment; col: emp (idx 0) and null replica index (idx 1 ~ n_rep).
    count_arr = np.zeros((3, n_rep + 1))
    count_arr[0, :] = np.sum(sent_arr == 1, axis=0)
    count_arr[1, :] = np.sum(sent_arr == 0, axis=0)
    count_arr[2, :] = np.sum(sent_arr == -1, axis=0)
    # Sum across sentiment dimension.
    ratio_mat = count_arr / np.sum(count_arr, axis=-2, keepdims=True)
    # Relative to null model: (empirical - null) / null.
    a = ratio_mat[..., 0:1] - ratio_mat[..., 1:]
    b = ratio_mat[..., 1:]
    res = a / b * 100  # In percentage.

    # When sample size is small, it's likely that there's no NEG in the samples (be it emp or null).
    # Then a/b could be ill-defined. Later in the process use nan operations.
    res[~np.isfinite(res)] = np.nan  # Convert inf to nan.
    return res


def get_sentiment_ratios(sent_emp, sent_nul, old_def=False):
    """
    Calculate sentiment ratio percentage change relative to 1 null replica.
    Args:
        sent_emp/null (1D np.arr): Each sample (citation pair) has a sentiment.
            emp: Empirical.
            nul: A null replica.

    Edge cases in (empirical - null) / null calculations:
    When sample size is small, it's likely that there's no NEG in the samples (be it emp or null).
        1) define 0/0 (np.nan by np default) as 0.
            If numerator is zero, res is 0 because percentage diff is 0, so no change.
        2) define non-zero/0 (np.inf by np default) as np.nan.
            If denominator is zero (numerator non-zero), res is np.nan because we can't do computational math with np.inf.
            Later we use nan operations, to skip these "inf" cases.
    """
    # Count sentiment for empirical and null.
    r_emp = np.array([np.sum(sent_emp == s) for s in [1, 0, -1]]) * 1.0
    r_nul = np.array([np.sum(sent_nul == s) for s in [1, 0, -1]]) * 1.0
    # Calculate ratio for each sentiment.
    r_emp /= np.sum(r_emp) * 1.0
    r_nul /= np.sum(r_nul) * 1.0
    # Relative to null model: (empirical - null) / null, in percentage.
    if old_def:
        res = (r_emp - r_nul) / r_nul * 100.0
        # When sample size is small, it's likely that there's no NEG in the samples (be it emp or null).
        # Then a/b could be ill-defined. Later in the process use nan operations.
        res[~np.isfinite(res)] = np.nan  # Convert inf to nan.
    else:
        num = (r_emp - r_nul) * 100.0
        res = np.where(num == 0, 0.0, np.where(r_nul == 0, np.nan, num / r_nul))  # Will still warn, but it's fine.
    return res


def find_y_rand_samp(cite2sent_emp, cite2sent_nul, pairs_to_sample, n_samp=None, n_rand_samp=1000, full_samp=True, old_def=False):
    """Bootstrap y (sentiment ratio percentage change rel. to null) with null stochasticity.
    Bootstrap from pairs_to_sample,
    each resample contains n_samp pairs, which have empirical sentiment and a null realization is sampled,
    then calculate y for each resample.
    This way it accounts for both sample size and null variance.

    Args:
        cite2sent_emp: One empirical sentiment per pair
        cite2sent_nul: null distribution (len-3 list/np.arr) per pair
        pairs_to_sample: Citation pairs to sample from.
        n_samp (int): Num of sample size in each sampling repetition. Default to len(pairs_to_sample).
        n_rand_samp (int): Num of random sampling repetition to estimate the std. Defaults to 1000.
        full_samp (bool): If True, return full samples (3 sentiment by n_rand_samp).
    """
    if len(pairs_to_sample) == 0:  # Nothing to sample from.
        return np.full((3, n_rand_samp), np.nan) if full_samp else [np.full(3, np.nan), np.full(3, np.nan)]
    rng = np.random.default_rng()
    if n_samp is None:
        n_samp = len(pairs_to_sample)
    samples = np.full((3, n_rand_samp), np.nan)
    # Sample citation pairs: n_samp x n_rand_samp times from the select pairs.
    indices = rng.choice(len(pairs_to_sample), size=(n_rand_samp, n_samp), replace=True)
    for i in range(n_rand_samp):
        # Sentiment for both empirical and null for each of the n_samp pairs.
        sent_emp = np.array([cite2sent_emp[pairs_to_sample[p]] for p in indices[i, :]])
        sent_nul = np.array([rng.choice([1, 0, -1], p=cite2sent_nul[pairs_to_sample[p]]) for p in indices[i, :]])
        # Calculate sentiment ratio percentage change relative to null model.
        samples[:, i] = get_sentiment_ratios(sent_emp, sent_nul, old_def=old_def)

    if full_samp:
        return samples
    else:
        return [np.nanmean(samples, axis=-1), np.nanstd(samples, axis=-1, ddof=1)]


def find_y_rand_samp_pre_agg(cite2sent_1, cite2sent_n, n_rand_samp=1000, full_samp=True, save_as_cite_dict=None):
    """
    a) vanilla (find_y_rand_samp()): null distribution is categorical on each pair.
        and because in vanilla, we find sentiment ratio of a group of pairs that consist different categorical distributions, we estimate the ratio by randomly sampling from the distributions.
    b) pre-agg: categorical distribution is on each citation sentence.
    For each pair:
        Say c is empirical count of a certain sentiment, x is random variable of sentiment count given null, and ns is constant (num of sentences).
        Let r_e=c/ns be empirical ratio, and r_n=x/ns be null ratio.
        then sentiment ratio percentage change rel. to null is: y = 100*(r_e - r_n)/r_n (same as vanilla).
        r_n follows multinomial distribution, but 1/r_n is not a standard distribution.
    Each null realization is number of sentiment for each sentiment. We use it to find y.

    Args:
        cite2sent_1: Empirical sentiments per pair, one for each citation sentence.
        cite2sent_n: null distribution (len-3 list/np.arr) per pair, but same categorical for each citation sentence.
        n_rand_samp (int): Num of random sampling repetition to draw from null. Defaults to 1000.
        save_as_cite_dict (dir): If provided, samples is in the form of a cite2 dict, and saved to this dir.
    """
    sent2idx = {1: 0, 0: 1, -1: 2}
    rng = np.random.default_rng()
    if save_as_cite_dict is not None:
        samples = {c: np.full((3, n_rand_samp), np.nan) for c in cite2sent_1}
    else:
        samples = np.full((len(cite2sent_1), 3, n_rand_samp), np.nan)
    for i, (c, sl) in tqdm(enumerate(cite2sent_1.items())):
        sc = np.zeros(3)  # Count of sentiment for 1, 0, -1, in this order.
        for s in sl:
            sc[sent2idx[s]] += 1
        # "/ len(sl)" to find sentiment ratio.
        r_emp = np.tile(sc, (n_rand_samp, 1)) / len(sl)  # dim=(n_rand_samp, 3)
        r_nul = rng.multinomial(len(sl), cite2sent_n[c], size=n_rand_samp) / len(sl)  # dim=(n_rand_samp, 3)
        # Calculate sentiment ratio percentage change relative to null model.
        num = (r_emp - r_nul) * 100.0
        tmp = np.where(num == 0, 0.0, np.where(r_nul == 0, np.nan, num / r_nul)).T  # dim=(3, n_rand_samp)
        if save_as_cite_dict is not None:
            samples[c] = tmp
        else:
            samples[i, ...] = tmp

    if full_samp:
        if save_as_cite_dict is not None:
            savePKL(save_as_cite_dict, "cite2SRPC", samples)
        else:
            return samples
    else:
        if save_as_cite_dict is not None:
            savePKL(
                save_as_cite_dict,
                "cite2SRPC_ms",
                {c: [np.nanmean(arr, axis=-1), np.nanstd(arr, axis=-1, ddof=1)] for c, arr in samples.items()},
            )
        else:
            return [np.nanmean(samples, axis=-1), np.nanstd(samples, axis=-1, ddof=1)]


def save_cite2t_collab(dir_dict, lag=0):
    """
    Concerning citations with collaborations (AKA cite with collab), there are 3 cases for a given citer-citee pair:
    "<" and "<=" are about the time of cite/collab:
    1) cite_a < 1st collab <= cite_b; n=450 before collab; n=581 after collab.
    2)          1st collab <= cite_b
    3) cite_a < 1st collab
    We can't properly test it with case 1) on a pair-by-pair basis since we effectively have no sample size of citations given a citer-citee pair; so the next best thing is to aggregate the LHS across all 3 cases and do the same for RHS and compare; this way we have way larger power, with the trade-off that the result is on the average level
    Comparing apples and oranges:

    cite2t_collab (dict):
        key: all but self-cite keys (citing-cited paper pairs)
        val: time since collaboration
            <0 means cite year < 1st collab year
            =0 means cite year = 1st collab year
            >0 means cite year > 1st collab year
    To find all citer-citee (last author) pairs in case 1):
        a) find all citer-citee (last author) pair (found by using all citing-cited (paper) pairs):
        b) if citer-citee pair has both of the below:
            <0 cite2t_collab val
            >=0 cite2t_collab val
        c) then this citer-citee is in case 1).
    Ultimately, we will use citing-cited (paper) pairs, not citer-citee (last author) pairs.
    So we have two sub-dictionaries:
    cite2t_collab_pre and cite2t_collab_post;
    both dictionaries concern the same bunch of citer-citee pairs (all that belong in case 1).

    """
    # Time to collaborate (-2 means 2 years before first collab in the dataset)
    g_coau_t = loadPKL(dir_dict, "g_coau_t")  # Time-varying coauthorship network.
    paper2year = loadPKL(dir_dict, "paper2year")  # Publication year for each paper.
    paper2last_author = loadPKL(dir_dict, "paper2last_author")  # Last author for each paper.
    cite2sent_2 = loadPKL(dir_dict, "cite2sent_2")  # For the citation pairs (keys) only.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # For finding and excluding self-cite.
    year_lim = [np.min(list(paper2year.values())), np.max(list(paper2year.values()))]
    cite2t_collab = {pair: None for pair in cite2sent_2 if cite2distance[pair] != 0}  # Skip self-cite.
    for pair in cite2sent_2:
        try:
            dic = g_coau_t.edges[paper2last_author[pair[0]], paper2last_author[pair[1]]]
        except KeyError:  # Never collaborate in the dataset.
            cite2t_collab[pair] = -np.inf
        else:  # Run if no exception (i.e., has collaborated at some point).
            for y in range(year_lim[0] + lag, year_lim[1] + lag + 1):  # +1 to include end point.
                if dic[y] == 1:
                    first = y - lag  # 1st collab (coauthor'ed) in year y-lag.
                    break
            else:  # Only run if break is not triggered, meaning something went wrong.
                raise Exception(f"DEBUG For citing-cited paper pair {pair}, first year of collaboration is not found!")
            cite2t_collab[pair] = paper2year[pair[0]] - first

    savePKL(dir_dict, "cite2t_collab", cite2t_collab)

