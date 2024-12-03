import networkx as nx
import numpy as np
import pickle
import os
import itertools
from tqdm import tqdm

from external_methods import process_batch_outputs
from helper_functions import cosine_sim, get_collab_distance


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
        using row2rate (in dir_TEMP) and sentrow2edgeinfo (in dir_TEMP),
        former of which contains values that are raw output from ChatGPT
        latter of which is a bookkeeping file.
    Save
    ----
    - cite2sent (dict):
        key: (pmcid_i, pmcid_j); citation edge
        val: []
            each member of the list is empirical sentiment of an instance of citation
            paper i can cite paper j multiple times
            this function saves cite2sent as "cite2sent_1.pkl" in dir_TEMP
    """
    row2rate = process_batch_outputs(dir_batch, dir_TEMP)
    with open(os.path.join(dir_TEMP, "sentrow2edgeinfo.pkl"), "rb") as f:
        sentrow2edgeinfo = pickle.load(f)
    with open(os.path.join(dir_TEMP, "cite2sent_0.pkl"), "rb") as f:
        cite2sent = pickle.load(f)
    if len(row2rate) != len(sentrow2edgeinfo):
        raise Exception(f"<row2rate> has {len(row2rate)} lines, but <sentrow2edgeinfo> has {len(sentrow2edgeinfo)} lines")
    for i in sentrow2edgeinfo.keys():
        citing_idx = int(sentrow2edgeinfo[i][1])
        cited_idxs = sentrow2edgeinfo[i][2]

        if isinstance(row2rate[i], str):
            rate = row2rate[i].strip()
        elif isinstance(row2rate[i], float):
            if row2rate[i].is_integer():
                rate = int(row2rate[i])
            else:
                raise Exception(f"{row2rate[i]} is a non-int float")
        elif isinstance(row2rate[i], int):
            rate = row2rate[i]
        else:
            raise Exception(f"{row2rate[i]} is a {type(row2rate[i])}")
        try:
            rate = int(rate)
        except ValueError:
            print(f"\nDEBUG unexpected ChatGPT response (row_idx={i}):\n{rate}")
            rate = 0  # default them to neutral
        except Exception as err:
            print(f"\nDEBUG other error: {err}")
            rate = 0  # default them to neutral
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
    with open(os.path.join(dir_dict, "cite2sent_2.pkl"), "rb") as f:
        cite2sent_2 = pickle.load(f)
    with open(os.path.join(dir_temp, "cite2ns.pkl"), "rb") as f:
        cite2ns = pickle.load(f)
    with open(os.path.join(dir_temp, "cite2title_sim.pkl"), "rb") as f:
        cite2title_sim = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2meta.pkl"), "rb") as f:
        paper2meta = pickle.load(f)

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
    ns_types = np.arange(1, maxN + 1)  # Only first 15 ns.
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

    with open(os.path.join(dir_dict, "cite2sent_null_param.pkl"), "wb") as f:
        pickle.dump(cite2sent_null_param, f)


def save_g_coau_t(dir_dict):
    """build a time-varying/dependent coauthorship network (undirected)
    If collaborated, then there's an edge.
    Edge attr is collaboration year y.
    It's cumulative, meaning at each year, it concerns all collaborated papers thus far, excluding this year
    e.g., collab network at year 2024 is network of collaborations upto and including 2023, but not 2024
    """
    with open(os.path.join(dir_dict, "paper2year.pkl"), "rb") as f:
        paper2year = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2author_s.pkl"), "rb") as f:
        paper2author_s = pickle.load(f)
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
        # print(f"Year {y} done.")

    # Then we go from most recent year to the oldest as we turn it into cumulative.
    # Because we loop through all collaboration edges, there will always be at least one instance of collaboration,
    # as in year_first won't remain np.inf.
    for aui, auj, d in tqdm(g_coau_t.edges(data=True)):
        # years_collab = np.zeros(year_max - year_min + 1, dtype=int)
        year_first = np.inf  # First/Earliest year that they have collaborated.
        for y, c in d.items():  # c is collaboration count at year y.
            # years_collab[y - year_min] = c
            if year_first > y:
                year_first = y
        year_first = int(year_first)
        for y in range(year_max + 1, year_first, -1):
            # shortest path is distance, collab times is inverse, so this is commented out, so is years_collab
            # g_coau_t.edges[aui, auj][y] = np.sum(years_collab[0 : y - year_min])
            g_coau_t.edges[aui, auj][y] = 1
        for y in range(year_first, year_min - 1, -1):
            g_coau_t.edges[aui, auj][y] = np.inf  # haven't collaborated yet

    with open(os.path.join(dir_dict, "g_coau_t.pkl"), "wb") as f:
        pickle.dump(g_coau_t, f)


def save_cite2distance(cite_pairs, dir_dict):
    """
    "distance" is collaboration distance
    which is shortest path length in collab network at the time of citation.
    cite_pairs is list-like of citation pair: (pmcid_i, pmcid_j)
    """
    with open(os.path.join(dir_dict, "g_coau_t.pkl"), "rb") as f:
        g_coau_t = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2last_author.pkl"), "rb") as f:
        paper2last_author = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2year.pkl"), "rb") as f:
        paper2year = pickle.load(f)
    # Manually create a subgraph view of each year's collab network.
    # In the earliest year, no one will have collaborate in the previous year yet since we don't have the data.
    # In the latest year, the collab network is 1 year after, but we wouldn't use it since it's beyond the data.
    year_min = min([x for x in paper2year.values()])
    year_max = max([x for x in paper2year.values()])
    if year_min != 1998 or year_max != 2023:
        raise Exception(f"paper2year range != [1998,2023], but {(year_min, year_max)}, change viewxxxx and this if condition accordingly")
    view1998 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][1998] == 1)  # all np.inf
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
    view2024 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2024] == 1)  # unused

    cite2CD = {e: np.nan for e in cite_pairs}
    for e in tqdm(cite_pairs):
        aui = paper2last_author[e[0]]
        auj = paper2last_author[e[1]]

        # most recent year + 1
        # cite2CD[e] = get_collab_distance(locals()[f"view{year_max + 1}"], aui, auj, weight=None)
        # year of citation
        cite2CD[e] = get_collab_distance(locals()[f"view{paper2year[e[0]]}"], aui, auj, weight=None)

    with open(os.path.join(dir_dict, "cite2distance.pkl"), "wb") as f:
        pickle.dump(cite2CD, f)


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


def get_sentiment_ratios(sent_emp, sent_nul):
    """
    Calculate sentiment ratio percentage change relative to 1 null replica.
    Args:
        sent_emp/null (1D np.arr): Each sample (citation pair) has a sentiment.
            emp: Empirical.
            nul: A null replica.
    """
    # Count sentiment for empirical and null.
    r_emp = np.array([np.sum(sent_emp == s) for s in [1, 0, -1]]) * 1.0
    r_nul = np.array([np.sum(sent_nul == s) for s in [1, 0, -1]]) * 1.0
    # Calculate ratio for each sentiment.
    r_emp /= np.sum(r_emp) * 1.0
    r_nul /= np.sum(r_nul) * 1.0
    # Relative to null model: (empirical - null) / null, in percentage.
    res = (r_emp - r_nul) / r_nul * 100.0
    # When sample size is small, it's likely that there's no NEG in the samples (be it emp or null).
    # Then a/b could be ill-defined. Later in the process use nan operations.
    res[~np.isfinite(res)] = np.nan  # Convert inf to nan.
    return res


def find_y_rand_samp(cite2sent_emp, cite2sent_nul, pairs_to_sample, n_samp=None, n_rand_samp=1000, full_samp=False):
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
        n_rand_samp (int): Num of random sampling repetition to estimate the std. Defaults to 100.
        full_samp (bool): If True, return full samples (3 sentiment by n_rand_samp).
    """
    rng = np.random.default_rng()
    if n_samp is None:
        n_samp = len(pairs_to_sample)
    samples = np.zeros((3, n_rand_samp))
    # Sample citation pairs: n_samp x n_rand_samp times from the select pairs.
    indices = rng.choice(len(pairs_to_sample), size=(n_rand_samp, n_samp), replace=True)
    for i in tqdm(range(n_rand_samp)):
        # Sentiment for both empirical and null for each of the n_samp pairs.
        sent_emp = np.array([cite2sent_emp[pairs_to_sample[p]] for p in indices[i, :]])
        sent_nul = np.array([rng.choice([1, 0, -1], p=cite2sent_nul[pairs_to_sample[p]]) for p in indices[i, :]])
        # Calculate sentiment ratio percentage change relative to null model.
        samples[:, i] = get_sentiment_ratios(sent_emp, sent_nul)

    if full_samp:
        return samples
    else:
        return [np.nanmean(samples, axis=-1), np.nanstd(samples, axis=-1)]


def save_cite2t_collab(dir_dict):
    # Time to collaborate (-2 means 2 years before first collab in the dataset)
    with open(os.path.join(dir_dict, "g_coau_t.pkl"), "rb") as f:
        g_coau_t = pickle.load(f)  # Time-varying coauthorship network.
    with open(os.path.join(dir_dict, "paper2year.pkl"), "rb") as f:
        paper2year = pickle.load(f)  # Publication year for each paper.
    with open(os.path.join(dir_dict, "paper2last_author.pkl"), "rb") as f:
        paper2last_author = pickle.load(f)  # Publication year for each paper.
    with open(os.path.join(dir_dict, "cite2sent_2.pkl"), "rb") as f:
        cite2sent_2 = pickle.load(f)  # For the citation pairs (keys) only.
    year_lim = [np.min(list(paper2year.values())), np.max(list(paper2year.values()))]
    cite2t_collab = {pair: None for pair in cite2sent_2}
    for pair in cite2sent_2:
        try:
            dic = g_coau_t.edges[paper2last_author[pair[0]], paper2last_author[pair[1]]]
        except KeyError:  # Never collaborate in the dataset.
            cite2t_collab[pair] = -np.inf
        else:  # Run if no exception (i.e., has collaborated at some point).
            for y in range(year_lim[0] + 1, year_lim[1] + 1):
                if dic[y] == 1:
                    first = y - 1  # collab in y-1 year
                    break
            cite2t_collab[pair] = paper2year[pair[0]] - first

    with open(os.path.join(dir_dict, "cite2t_collab.pkl"), "wb") as f:
        pickle.dump(cite2t_collab, f)
