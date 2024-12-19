import numpy as np
import os

from cite_coauthor_functions import find_y_rand_samp
from helper_functions import reverse_dict_list, reverse_dict_val, savePKL, loadPKL


def prepare_collab_groups(dir_dict, dir_npy, n_bs=1000):
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.

    n1 = 4  # 4 distance types; [1, inf), [1], [2, inf), [0].
    n2 = 3  # 3 sentiment.
    # Calculate sentiment ratio percentage change relative to null model.
    ratio_mat_rel = np.zeros((n1, n2, n_bs))
    # Find all citaitons pairs of distance = [1, inf), [1], [2, inf), [0].
    pairs_list = [[pair for pair, v in cite2distance.items() if v >= 1]]
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 1])
    pairs_list.append([pair for pair, v in cite2distance.items() if v >= 2])
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 0])

    for idx, pairs in enumerate(pairs_list):
        # Sampling distribution of sentiment ratio percentage change for current country.
        ratio_mat_rel[idx, ...] = find_y_rand_samp(cite2sent_emp, cite2sent_nul, pairs, len(pairs), n_rand_samp=n_bs, full_samp=True)

    np.save(os.path.join(dir_npy, "ratio_mat_rel-collab_groups.npy"), ratio_mat_rel)


def prepare_collab_distance(dir_dict, dir_npy, dist_max=6, n_bs=1000):
    """Calculate sentiment ratio percentage change relative to null model.
    Args:
        dir_dict (str): Folder holding required data.
        dist_max (int): Max collab distance to plot.
        n_bs (int): Number of bootstrap resamples. Defaults to 1000.
    """
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.

    # row: Distance 0,1,2,...,dist_max; col: 3 sentiment; dep: n_bs data points.
    ratio_mat_rel = np.zeros((dist_max + 1, 3, n_bs))
    for d in range(dist_max + 1):
        pairs_to_sample = [k for k, v in cite2distance.items() if v == d]
        n_samp = len(pairs_to_sample)
        print(f"Bootstrapping: sample size {n_samp}, {n_bs} resamples at distance {d}...")
        ratio_mat_rel[d, ...] = find_y_rand_samp(cite2sent_emp, cite2sent_nul, pairs_to_sample, n_rand_samp=n_bs, full_samp=True)

    np.save(os.path.join(dir_npy, "ratio_mat_rel-collab_dist.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-collab_dist.npy"), np.arange(dist_max + 1))


def prepare_t_collab(dir_dict, dir_npy, year_ranges=None, n_bs=1000):
    """
    Args:
        year_ranges (list of tuples): t2collab range for each bin.
    """
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2t_collab = loadPKL(dir_dict, "cite2t_collab")  # Time before first collab in data.

    if year_ranges is None:
        year_ranges = [(-4, -3), (-2, -1), (0, 0), (1, 2), (3, 4), (5, 6)]
    n1 = len(year_ranges)  # Num of t2collab bins.
    n2 = 3  # 3 sentiment.
    # Calculate sentiment ratio percentage change relative to null model.
    ratio_mat_rel = np.zeros((n1, n2, n_bs))

    # Find corresponding sentiments (both empirical and null) for each bin.
    for idx, yr in enumerate(year_ranges):
        # Find citation pairs in the bin.
        pairs_sample = [p for p, t in cite2t_collab.items() if yr[0] <= t <= yr[1]]
        # Sampling distribution of sentiment ratio percentage change for each bin.
        ratio_mat_rel[idx, ...] = find_y_rand_samp(
            cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True
        )

    np.save(os.path.join(dir_npy, "ratio_mat_rel-t_collab.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-t_collab.npy"), np.array(year_ranges))

    ##### Same as before except this is for citations that never collaborated in the dataset.
    pairs_sample = [p for p, t in cite2t_collab.items() if np.isinf(t)]
    ratio_mat_rel = find_y_rand_samp(cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True)
    np.save(os.path.join(dir_npy, "ratio_mat_rel-t_collab_no_collab.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-t_collab_no_collab.npy"), np.inf)

    ##### Same as before except this is for citations that will collaborate in next 4 years in the dataset.
    pairs_sample = [p for p, t in cite2t_collab.items() if -4 <= t <= -1]
    ratio_mat_rel = find_y_rand_samp(cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True)
    np.save(os.path.join(dir_npy, "ratio_mat_rel-t_collab_will_collab.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-t_collab_will_collab.npy"), np.array([(-4, -1)]))


def prepare_hindex(dir_dict, dir_npy, binW=30, n_bs=1000):
    """
    Args:
        binW (int): Bin width on either side of 0 h-Index diff. Defaults to 30.
    """
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author = loadPKL(dir_dict, "paper2last_author")  # Last author name of each paper.
    last_author2hIndex = loadPKL(dir_dict, "last_author2hIndex")  # Last author hIndex.

    def _is_hIndex_in_range(paper_i, paper_j, hidx_low, hidx_high, right_inclusive=False):
        author_i = paper2last_author[paper_i]
        author_j = paper2last_author[paper_j]
        if author_i not in last_author2hIndex or author_j not in last_author2hIndex:
            return False
        # h-Index difference: citing - cited.
        hidx_diff = last_author2hIndex[author_i] - last_author2hIndex[author_j]
        if right_inclusive:
            return hidx_low <= hidx_diff <= hidx_high
        else:
            return hidx_low <= hidx_diff < hidx_high

    bins = np.concatenate([np.arange(-binW / 2, -100, -binW)[::-1], [binW / 2], np.arange(binW / 2, 100, binW)[1:]])
    print(f"h-Index bin boundaries: {bins}.")
    n1 = 3  # 3 distance types; [1, inf), [1], [2, inf).
    n2 = len(bins) - 1  # Num of h-Index bins.
    n3 = 3  # 3 sentiment.
    # Calculate sentiment ratio percentage change relative to null model.
    ratio_mat_rel = np.zeros((n1, n2, n3, n_bs))

    # Find all citaitons pairs of distance = [1, inf), [1], and [2, inf).
    pairs_list = [[pair for pair, v in cite2distance.items() if v >= 1]]
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 1])
    pairs_list.append([pair for pair, v in cite2distance.items() if v >= 2])

    for idx, pairs in enumerate(pairs_list):
        # Find corresponding sentiments (both empirical and null) for each bin.
        for b in range(n2):  # Loop through each bin.
            # Find citation pairs in the bin.
            pairs_sample = [
                p for p in pairs if _is_hIndex_in_range(p[0], p[1], bins[b], bins[b + 1], right_inclusive=(b == n2 - 1))
            ]  # Right inclusive if last bin.
            # Sampling distribution of sentiment ratio percentage change for each bin.
            ratio_mat_rel[idx, b, ...] = find_y_rand_samp(
                cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True
            )

    np.save(os.path.join(dir_npy, "ratio_mat_rel-hindex.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-hindex.npy"), bins)


def prepare_country_effects(dir_dict, dir_npy, n_bs=1000, thres=100):
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author_country = loadPKL(dir_dict, "paper2last_author_country")  # Last author countries for each paper.
    last_author_country2paper = reverse_dict_list(paper2last_author_country)
    n_collab = {c: 0 for c in last_author_country2paper.keys()}
    for e, sent in cite2sent_emp.items():
        if cite2distance[e] == 1:
            for c in paper2last_author_country[e[0]]:
                n_collab[c] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    n_collab = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    countries_select = np.array(list(n_collab.keys()))

    n1 = 3  # 3 distance types; [1, inf), [1], [2, inf).
    n2 = len(countries_select)  # Num of countries selected.
    n3 = 3  # 3 sentiment.
    # Calculate sentiment ratio percentage change relative to null model.
    ratio_mat_rel = np.zeros((n1, n2, n3, n_bs))

    # Find all citaitons pairs of distance = [1, inf), [1], and [2, inf).
    pairs_list = [[pair for pair, v in cite2distance.items() if v >= 1]]
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 1])
    pairs_list.append([pair for pair, v in cite2distance.items() if v >= 2])

    for idx, pairs in enumerate(pairs_list):
        # Find the multi-hot country arr for these pairs.
        country_arr = np.array([np.in1d(countries_select, paper2last_author_country[pair[0]], assume_unique=True) for pair in pairs])
        # Find corresponding sentiments (both empirical and null) for each country.
        for c, _ in enumerate(countries_select):
            # pairs indices where citing paper whose last author is in country countries_select[c].
            indices = country_arr[:, c].nonzero()[0]
            pairs_sample = [pairs[i] for i in indices]
            # Sampling distribution of sentiment ratio percentage change for current country.
            ratio_mat_rel[idx, c, ...] = find_y_rand_samp(
                cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True
            )

    np.save(os.path.join(dir_npy, "ratio_mat_rel-country_effect.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-country_effect.npy"), countries_select)


def prepare_department_effects(dir_dict, dir_npy, n_bs=1000, thres=100):
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author_department = loadPKL(dir_dict, "paper2last_author_department_28_dep")  # Last author departments for each paper.
    last_author_department2paper = reverse_dict_list(paper2last_author_department)
    n_collab = {c: 0 for c in last_author_department2paper.keys()}
    for e, sent in cite2sent_emp.items():
        if cite2distance[e] == 1:
            for c in paper2last_author_department[e[0]]:
                n_collab[c] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    n_collab = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    departments_select = np.array(list(n_collab.keys()))

    n1 = 3  # 3 distance types; [1, inf), [1], [2, inf).
    n2 = len(departments_select)  # Num of departments selected.
    n3 = 3  # 3 sentiment.
    # Calculate sentiment ratio percentage change relative to null model.
    ratio_mat_rel = np.zeros((n1, n2, n3, n_bs))

    # Find all citaitons pairs of distance = [1, inf), [1], and [2, inf).
    pairs_list = [[pair for pair, v in cite2distance.items() if v >= 1]]
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 1])
    pairs_list.append([pair for pair, v in cite2distance.items() if v >= 2])

    for idx, pairs in enumerate(pairs_list):
        # Find the multi-hot department arr for these pairs.
        department_arr = np.array(
            [np.in1d(departments_select, paper2last_author_department[pair[0]], assume_unique=True) for pair in pairs]
        )
        # Find corresponding sentiments (both empirical and null) for each department.
        for c, _ in enumerate(departments_select):
            # pairs indices where citing paper whose last author is in department departments_select[c].
            indices = department_arr[:, c].nonzero()[0]
            pairs_sample = [pairs[i] for i in indices]
            # Sampling distribution of sentiment ratio percentage change for current department.
            ratio_mat_rel[idx, c, ...] = find_y_rand_samp(
                cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True
            )

    np.save(os.path.join(dir_npy, "ratio_mat_rel-department_effect.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-department_effect.npy"), departments_select)


def prepare_gender_effects(dir_dict, dir_npy, n_bs=1000, thres=100):
    cite2sent_emp = loadPKL(dir_dict, "cite2sent_2")  # Each citation pair has 1 empirical sentiment.
    cite2sent_nul = loadPKL(dir_dict, "cite2sent_null_param")  # Null model sentiment.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author = loadPKL(dir_dict, "paper2last_author")
    last_author2gender_info = loadPKL(dir_dict, "last_author2gender_info")

    paper2last_author_gender = {p: last_author2gender_info[au] for p, au in paper2last_author.items()}
    paper2last_author_gender = {p: g[0] if (g[1] >= 0.7 and g[2] >= 20) else None for p, g in paper2last_author_gender.items()}

    last_author_gender2paper = reverse_dict_val(paper2last_author_gender)
    last_author_gender2paper.pop(None)  # None is ones didn't pass the filter earlier.
    n_collab = {c: 0 for c in last_author_gender2paper.keys()}
    for pair, d in cite2distance.items():
        ge = paper2last_author_gender[pair[0]]
        if d == 1 and ge is not None:
            n_collab[ge] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    n_collab = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    genders_select = np.array(list(n_collab.keys()))

    n1 = 4  # 4 distance types; [1, inf), [1], [2, inf), [0].
    n2 = len(genders_select)  # Num of genders selected.
    n3 = 3  # 3 sentiment.
    # Calculate sentiment ratio percentage change relative to null model.
    ratio_mat_rel = np.zeros((n1, n2, n3, n_bs))

    # Find all citaitons pairs of distance = [1, inf), [1], [2, inf), [0].
    pairs_list = [[pair for pair, v in cite2distance.items() if v >= 1]]
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 1])
    pairs_list.append([pair for pair, v in cite2distance.items() if v >= 2])
    pairs_list.append([pair for pair, v in cite2distance.items() if v == 0])

    for idx, pairs in enumerate(pairs_list):
        # Find the one-hot gender arr for these pairs.
        gender_arr = np.array([np.in1d(genders_select, [paper2last_author_gender[pair[0]]], assume_unique=True) for pair in pairs])
        # Find corresponding sentiments (both empirical and null) for each gender.
        for c, _ in enumerate(genders_select):
            # pairs indices where citing paper whose last author is in gender genders_select[c].
            indices = gender_arr[:, c].nonzero()[0]
            pairs_sample = [pairs[i] for i in indices]
            # Sampling distribution of sentiment ratio percentage change for current gender.
            ratio_mat_rel[idx, c, ...] = find_y_rand_samp(
                cite2sent_emp, cite2sent_nul, pairs_sample, len(pairs_sample), n_rand_samp=n_bs, full_samp=True
            )

    np.save(os.path.join(dir_npy, "ratio_mat_rel-gender_effect.npy"), ratio_mat_rel)
    np.save(os.path.join(dir_npy, "groups-gender_effect.npy"), genders_select)


def get_sample_size_department(dir_dict, thres=100):
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author_department = loadPKL(dir_dict, "paper2last_author_department_28_dep")  # Last author departments for each paper.
    last_author_department2paper = reverse_dict_list(paper2last_author_department)
    n_collab = {c: 0 for c in last_author_department2paper.keys()}
    for e, d in cite2distance.items():
        if d == 1:
            for c in paper2last_author_department[e[0]]:
                n_collab[c] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    n_collab = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    return n_collab


def get_sample_size_country(dir_dict, thres=100):
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author_country = loadPKL(dir_dict, "paper2last_author_country")# Last author departments for each paper.
    last_author_country2paper = reverse_dict_list(paper2last_author_country)
    n_collab = {c: 0 for c in last_author_country2paper.keys()}
    for e, d in cite2distance.items():
        if d == 1:
            for c in paper2last_author_country[e[0]]:
                n_collab[c] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    n_collab = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    return n_collab
