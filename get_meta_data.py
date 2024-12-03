"""
get processed data and meta data
"""

import os
import pickle
from tqdm import tqdm
import networkx as nx
from pubmed_parser import parse_pubmed_xml

"""
    ########################################
        Make lookup dictionaries
    ########################################
"""


def make_edges_and_meta(path_out, ref_stats, key_info_all):
    """
    Intermediaries
    --------------
    - refid2pmcid_lookup (dict):
        key is a refid of a reference node of current node;
        this ref node has to be in network too
        val is pmcid (int) in-network
        so this dict is different in each iteration finding neighbors for each node in network

    Return
    -----
    - "article_meta" (dict):
        key is pmcid (int)
        val (dict):
            article-type, jyf, title, pub-date
            authors (nested dict):
                key is author index (author order)
                value (author) is a len-3 dict of str
    - cnets (dict of 2D np.arr and dict):
        "sents" (dict of dict):  # in-network only
            lv-1 key is the sentences' paper's pmcid (int)
            key is idx for sents (nonce; never used outside of sents/cites dict)
                only have sents that are in-network
            val is str of citation sentence (1 cite_marker for each chain)
        "cites" (dict of dict):  # in-network only
            lv-1 key is the sentences' paper's pmcid (int)
            key is idx for sents (nonce; never used outside of sents/cites dict)
            val is list of sets (1 set for each cite_marker);
                val in the set is pmcid (int), or None if not
        "edges" (set of len-2 tup: (pmcidi, pmcidj)): citation network - edgelist
            only considers cited that are within the same network
            node: each XML in the "xml" folder
            if (pmcidi, pmcidj) in edge, then there's at least a citation from ni (pmcid=pmcidi) to nj (pmcid=pmcidj)
    """
    # network data variables (not neighbors)
    n = len(ref_stats)  # size of network
    pmcids = set(ref_stats.keys())  # pmcids (int; they are from key_info_all.keys()) of network nodes
    sents, cites = dict(), dict()
    sent_idx = -1  # count number of citation sentences in-network; start from 0
    article_meta = dict()

    # Construct networks.
    edges = set()
    for key, val in tqdm(ref_stats.items()):  # Find in-network neighbors for every node (in pmcid).
        pmcid = int(key)
        article_meta[pmcid] = {
            "article-type": key_info_all[key]["article-type"],
            "jyf": ref_stats[key]["jyf"],
            "title": key_info_all[key]["article-title"],
            "pub-date": key_info_all[key]["pub-date"],
            "authors": key_info_all[key]["authors"],
        }
        refid2pmcid_lookup = {
            r: int(val["ref-list"][r]["pub-id-pmcid"])
            for r in val["ref_within"]
            if int(val["ref-list"][r]["pub-id-pmcid"]) in pmcids  # exclude nodes from outside of dataset
        }
        # citation content network edge construction
        # unlike citation network that goes by ref-list (bibliography), this one is based on citation sentences
        for pmcid_ref in refid2pmcid_lookup.values():
            edges.add((pmcid, pmcid_ref))
        for kkey in key_info_all[key]["sents"].keys():  # loop over all sents in paper
            flag_inNetCite = False  # True if sent has at least 1 citation in-network
            for ks in key_info_all[key]["cites"][kkey]:  # ks is set, which has ref id for the chain
                for kr in ks:  # kr is ref id for the chain
                    if refid2pmcid_lookup.get(kr):  # find refid in-network; None if not
                        flag_inNetCite = True  # found a refid in-network, so sent is in-network
                        sent_idx += 1
                        if sents.get(pmcid):
                            sents[pmcid].update({sent_idx: key_info_all[key]["sents"][kkey]})
                        else:
                            sents[pmcid] = {sent_idx: key_info_all[key]["sents"][kkey]}
                        break
                if flag_inNetCite:
                    break
            if flag_inNetCite:  # make cites such that all in-network refs are pmcid (int) in-network
                if cites.get(pmcid):
                    cites[pmcid].update({sent_idx: [None] * len(key_info_all[key]["cites"][kkey])})
                else:
                    cites[pmcid] = {sent_idx: [None] * len(key_info_all[key]["cites"][kkey])}
                for j, ks in enumerate(key_info_all[key]["cites"][kkey]):
                    cites[pmcid][sent_idx][j] = {refid2pmcid_lookup.get(kr) for kr in ks}

    cnets = dict(edges=edges, sents=sents, cites=cites)
    with open(os.path.join(path_out, "article_meta.pkl"), "wb") as f:
        pickle.dump(article_meta, f)
    with open(os.path.join(path_out, "cnets2_Neuroscience.pkl"), "wb") as f:
        pickle.dump(cnets, f)
    # return article_meta, cnets


def save_paper_author_dicts(paper2meta, dir_dict):
    paper2author = {p: [d["authors"][i]["name"] for i in range(len(d["authors"]))] for p, d in paper2meta.items()}
    paper2author_s = {p: set(aus) for p, aus in paper2author.items()}
    paper2last_author = {p: aus[-1] for p, aus in paper2author.items()}
    paper2first_author = {p: aus[0] for p, aus in paper2author.items()}
    with open(os.path.join(dir_dict, "paper2author.pkl"), "wb") as f:
        pickle.dump(paper2author, f)
    with open(os.path.join(dir_dict, "paper2author_s.pkl"), "wb") as f:
        pickle.dump(paper2author_s, f)
    with open(os.path.join(dir_dict, "paper2last_author.pkl"), "wb") as f:
        pickle.dump(paper2last_author, f)
    with open(os.path.join(dir_dict, "paper2first_author.pkl"), "wb") as f:
        pickle.dump(paper2first_author, f)


def save_paper_time_dicts(paper2meta, dir_dict):
    paper2year = {p: m["jyf"][1] for p, m in paper2meta.items()}
    with open(os.path.join(dir_dict, "paper2year.pkl"), "wb") as f:
        pickle.dump(paper2year, f)


def save_aff_dicts(paper2meta, dir_dict):
    """Make author2aff dicts;
    unused because in the cases (i.e., country, department) where aff is involved,
    paper2meta is directly used to parse instead of using the results here.

    Args
    ----
    - paper2meta (dict): meta info for each paper
    - dir_dict (str): directory to save aff dicts to

    Save
    ----
    - last_author2aff (dict):
        key: len-2 tuple of author name
        val: set of affiliation strings
    """
    first_author2aff = dict()
    last_author2aff = dict()
    for p, m in tqdm(paper2meta.items()):
        first_author = m["authors"][0]
        last_author = m["authors"][len(m["authors"]) - 1]

        if first_author["name"] not in first_author2aff:
            first_author2aff[first_author["name"]] = set()
        if last_author["name"] not in last_author2aff:
            last_author2aff[last_author["name"]] = set()

        first_author2aff[first_author["name"]].add(first_author["aff"])
        last_author2aff[last_author["name"]].add(last_author["aff"])

    with open(os.path.join(dir_dict, "first_author2aff.pkl"), "wb") as f:
        pickle.dump(first_author2aff, f)
    with open(os.path.join(dir_dict, "last_author2aff.pkl"), "wb") as f:
        pickle.dump(last_author2aff, f)


def save_and_parse_full_titles(dir_xml, dir_dict):
    """use parse_pubmed_xml() full_title to get paper titles.
    In parse_pubmed_xml(): pub_year = int(pub_date_dict["year"]) line may yield KeyError: "year".

    Args:
        dir_xml: Folder holding xml files.
        dir_dict: Folder holding paper2meta.
    """
    with open(os.path.join(dir_dict, "paper2meta.pkl"), "rb") as f:
        paper2meta = pickle.load(f)
    # Title from our parser.
    paper2title = {p: v["title"] for p, v in paper2meta.items()}
    paper2full_title = {p: None for p, v in paper2meta.items()}
    # Title from pubmed_parser package parser.
    files = set(file for file in os.listdir(dir_xml) if file.endswith(".xml"))
    for p in tqdm(paper2meta):
        tmp1 = f"{p}.xml"
        if tmp1 not in files:
            tmp1 = f"PMC{p}.xml"
        if tmp1 not in files:
            raise Exception(f"paper PMC={p} is not in {dir_xml}.")
        tmp1 = os.path.join(dir_xml, tmp1)
        dict_ = parse_pubmed_xml(tmp1)
        tmp_ = dict_.get("full_title")
        if tmp_ is None:
            print(f"paper PMC={p} full_title not found by parse_pubmed_xml().")
        else:
            paper2full_title[p] = tmp_

    with open(os.path.join(dir_dict, "paper2title.pkl"), "wb") as f:
        pickle.dump(paper2title, f)
    with open(os.path.join(dir_dict, "paper2full_title.pkl"), "wb") as f:
        pickle.dump(paper2full_title, f)
