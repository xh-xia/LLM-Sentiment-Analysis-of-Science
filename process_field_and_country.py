import os
import networkx as nx
from tqdm import tqdm
import numpy as np
import pandas as pd
import regex as re
from country_list import countries_for_language

from xml_parser import DASH
from helper_functions import fuzz_check, reverse_dict_list, has_keyword_any, savePKL, loadPKL, flatten_list


def get_country(affiliation, country_name2country_abbr, set_us_state, all_names_re, all_US_states_re):


    # Last part of aff string
    res = re.split(", |\n", affiliation)
    tmp1 = res[-1].strip()
    if tmp1.endswith(".") or tmp1.endswith(";"):
        tmp1 = tmp1[:-1]

    # 1) Match full country name or full US state name.
    if tmp1 in country_name2country_abbr:
        return country_name2country_abbr[tmp1], 100
    elif tmp1 in set_us_state:  # write out US state name in full
        return country_name2country_abbr["USA"], 100
    elif tmp1.split(" ")[-1] in country_name2country_abbr:
        return country_name2country_abbr[tmp1.split(" ")[-1]], 100


    # 2) Match USA state (most XX zip-code ones seem to be in USA).
    # Only for those in "XX+ 12345" or "XX+ 12345-6789" format, which may or may not follow other strings that follow a comma
    # res2 = re.search(rf"([A-Za-z\s]\s)*?({all_US_states_re})\s(\d{5}([{DASH}]\d{4})?)$", tmp1)
    res2 = re.search(rf".*?({all_US_states_re})\s(\d\d\d\d\d([{DASH}]\d\d\d\d)?)$", tmp1)
    if res2 is not None:
        # if res2.group(2) in set_us_state_abbr or res2.group(2) in set_us_state or res2.group(2) in country_abbr2country_name["US"]:
        return country_name2country_abbr["USA"], 100

    # 3) Full string search of full country name.
    res3 = re.search(rf".*?({all_names_re}).*?", affiliation)
    if res3 is not None:
        return country_name2country_abbr[res3.group(1)], 100

    # 4) Last matching method, try each part of the full string.
    best_name = 0
    best_similarity = 0

    for name in country_name2country_abbr:
        for w in re.split(", |\n", affiliation):
            fc = fuzz_check(w, name)
            if fc > best_similarity:
                best_similarity = fc
                best_name = name
    if best_name != 0:
        return country_name2country_abbr[best_name], best_similarity
    else:
        return None, 0


def save_country_dicts(paper2meta, dir_out, print_fail=False):

    # Set up constants.
    # 57 US states and territories. They often don't write country, which is why we include the states for US
    set_us_state = {"Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
    "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming","District of Columbia","American Samoa","Guam","Northern Mariana Islands","Puerto Rico","United States Minor Outlying Islands", "Virgin Islands, U.S."}
    set_us_state_abbr = {"AL", "KY", "OH","AK", "LA", "OK","AZ", "ME", "OR","AR", "MD", "PA","AS", "MA", "PR","CA", "MI", "RI","CO", "MN", "SC","CT", "MS", "SD","DE", "MO", "TN","DC", "MT","TX","FL", "NE", "TT","GA", "NV", "UT","GU", "NH", "VT","HI", "NJ", "VA","ID", "NM", "VI","IL", "NY", "WA","IN", "NC", "WV","IA", "ND", "WI","KS", "MP", "WY"}

    # Use country_list.countries_for_language library to get country names from 2-letter codes.
    # But it contains American Samoa, Northern Mariana Islands, Puerto Rico, Guam that are in set_us_state
    # "Georgia" is the name for both the country and US state, but abbreviations are different.

    country_abbr2country_name = {k: {v} for k, v in dict(countries_for_language("en")).items() if v not in set_us_state.difference({"Georgia"})}

    country_abbr2country_name["US"].add("USA")
    country_abbr2country_name["US"].add("U.S.A.")
    country_abbr2country_name["US"].add("U.S.A")
    country_abbr2country_name["CN"].add("People's Republic of China")
    country_abbr2country_name["CN"].add("PRC")
    country_abbr2country_name["CN"].add("PRoC")
    country_abbr2country_name["GB"].add("UK")
    country_abbr2country_name["GB"].add("U.K.")
    country_abbr2country_name["GB"].add("England")
    country_abbr2country_name["GB"].add("United Kingodm")
    country_abbr2country_name["GB"].add("Great Britain")
    country_abbr2country_name["TR"].add("Türkiye")
    country_abbr2country_name["CZ"].add("Czech Republic")
    country_abbr2country_name["SK"].add("Slovak Republic")
    country_abbr2country_name["KR"].add("Republic of Korea")
    country_abbr2country_name["KR"].add("Korea South")
    country_abbr2country_name["KR"].add("Korea")
    country_abbr2country_name["MX"].add("México")
    country_abbr2country_name["CH"].add("Suisse")

    country_name2country_abbr = reverse_dict_list(country_abbr2country_name)
    for name, abbr in country_name2country_abbr.items():
        if len(abbr) != 1:
            print(name, abbr)
            raise
    country_name2country_abbr = {k:v[0] for k,v in country_name2country_abbr.items()}
    
    all_names_re = "|".join(country_name2country_abbr).replace(".", "\.")  # escape period char
    all_US_states_re = "|".join(set_us_state_abbr.union(set_us_state).union(country_abbr2country_name["US"])).replace(".", "\.")
    
    # Parse countries.
    paper2last_author_country = {p: set() for p in paper2meta}
    paper2first_author_country = {p: set() for p in paper2meta}

    success = 0
    fail = 0  # lower than 1.0 of fuzz_check is considered a fail

    for p in paper2meta.keys():
        idx = len(paper2meta[p]["authors"]) - 1  # Last author.
        pap = paper2meta[p]["authors"][idx]["aff"]
        for aff in pap[1]:
            aff = aff.strip()
            a, b = get_country(aff, country_name2country_abbr, set_us_state, all_names_re, all_US_states_re)
            if b != 100:
                # print(re.split(", |\n", aff))
                if print_fail:
                    print(p, aff)
                fail += 1
            else:
                paper2last_author_country[p].add(a)
                success += 1

        pap = paper2meta[p]["authors"][0]["aff"]  # First author.
        for aff in pap[1]:
            aff = aff.strip()
            a, b = get_country(aff, country_name2country_abbr, set_us_state, all_names_re, all_US_states_re)
            if b != 100:
                if print_fail:
                    print(p, aff)
                fail += 1
            else:
                paper2first_author_country[p].add(a)
                success += 1
    print(f"success: {success} | fail: {fail} | fail rate {fail / (fail + success)*100:.2f}%")

    paper2last_author_country = {k: list(v) for k, v in paper2last_author_country.items()}
    paper2first_author_country = {k: list(v) for k, v in paper2first_author_country.items()}

    savePKL(dir_out, "paper2last_author_country", paper2last_author_country)
    savePKL(dir_out, "paper2first_author_country", paper2first_author_country)


def save_country_dicts_all_authors(paper2meta, dir_out, print_fail=False):

    # Set up constants.
    # 57 US states and territories. They often don't write country, which is why we include the states for US
    set_us_state = {"Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
    "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming","District of Columbia","American Samoa","Guam","Northern Mariana Islands","Puerto Rico","United States Minor Outlying Islands", "Virgin Islands, U.S."}
    set_us_state_abbr = {"AL", "KY", "OH","AK", "LA", "OK","AZ", "ME", "OR","AR", "MD", "PA","AS", "MA", "PR","CA", "MI", "RI","CO", "MN", "SC","CT", "MS", "SD","DE", "MO", "TN","DC", "MT","TX","FL", "NE", "TT","GA", "NV", "UT","GU", "NH", "VT","HI", "NJ", "VA","ID", "NM", "VI","IL", "NY", "WA","IN", "NC", "WV","IA", "ND", "WI","KS", "MP", "WY"}

    # Use country_list.countries_for_language library to get country names from 2-letter codes.
    # But it contains American Samoa, Northern Mariana Islands, Puerto Rico, Guam that are in set_us_state
    # "Georgia" is the name for both the country and US state, but abbreviations are different.

    country_abbr2country_name = {k: {v} for k, v in dict(countries_for_language("en")).items() if v not in set_us_state.difference({"Georgia"})}

    country_abbr2country_name["US"].add("USA")
    country_abbr2country_name["US"].add("U.S.A.")
    country_abbr2country_name["US"].add("U.S.A")
    country_abbr2country_name["CN"].add("People's Republic of China")
    country_abbr2country_name["CN"].add("PRC")
    country_abbr2country_name["CN"].add("PRoC")
    country_abbr2country_name["GB"].add("UK")
    country_abbr2country_name["GB"].add("U.K.")
    country_abbr2country_name["GB"].add("England")
    country_abbr2country_name["GB"].add("United Kingodm")
    country_abbr2country_name["GB"].add("Great Britain")
    country_abbr2country_name["TR"].add("Türkiye")
    country_abbr2country_name["CZ"].add("Czech Republic")
    country_abbr2country_name["SK"].add("Slovak Republic")
    country_abbr2country_name["KR"].add("Republic of Korea")
    country_abbr2country_name["KR"].add("Korea South")
    country_abbr2country_name["KR"].add("Korea")
    country_abbr2country_name["MX"].add("México")
    country_abbr2country_name["CH"].add("Suisse")

    country_name2country_abbr = reverse_dict_list(country_abbr2country_name)
    for name, abbr in country_name2country_abbr.items():
        if len(abbr) != 1:
            print(name, abbr)
            raise
    country_name2country_abbr = {k:v[0] for k,v in country_name2country_abbr.items()}
    
    all_names_re = "|".join(country_name2country_abbr).replace(".", "\.")  # escape period char
    all_US_states_re = "|".join(set_us_state_abbr.union(set_us_state).union(country_abbr2country_name["US"])).replace(".", "\.")
    
    # Parse countries.
    paper2all_author_country = {p: [set() for _ in paper2meta[p]["authors"]] for p in paper2meta}

    success = 0
    fail = 0  # lower than 1.0 of fuzz_check is considered a fail
    for p in paper2meta:
        aus_dict = paper2meta[p]["authors"]
        if len(aus_dict) == 0:  # No authors found, skip.
            print(f"PMC={p} is skipped because there are no authors found.")
            continue
        for au_order in aus_dict.keys():  # Do it for each author on the author list.
            affTup = aus_dict[au_order]["aff"]  # author aff info tuple ("name": last author name; "aff": tuple).
            for aff in affTup[1]:  # affTup[1] is a tuple of aff str.
                aff = aff.strip()
                a, b = get_country(aff, country_name2country_abbr, set_us_state, all_names_re, all_US_states_re)
                if b != 100:
                    # print(re.split(", |\n", aff))
                    if print_fail:
                        print(p, aff)
                    fail += 1
                else:
                    paper2all_author_country[p][au_order].add(a)
                    success += 1

    print(f"success: {success} | fail: {fail} | fail rate {fail / (fail + success)*100:.2f}%")

    paper2all_author_country = {k: [list(x) for x in v] for k, v in paper2all_author_country.items()}

    savePKL(dir_out, "paper2all_author_country", paper2all_author_country)


def save_country_measures(dir_6d, dir_dict, thres=50):
    """countries is list-like containing 2-letter code of countries to be used in analysis
    thres (int):
        Num of citation pairs at distance 1 needed to consider a country (by last author affiliation countries)
    """
    paper2last_author_country = loadPKL(dir_dict, "paper2last_author_country")
    paper2last_author = loadPKL(dir_dict, "paper2last_author")
    last_author2gender_info = loadPKL(dir_dict, "last_author2gender_info")
    # Select subset of countries based on citations.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    last_author_country2paper = reverse_dict_list(paper2last_author_country)
    n_collab = {c: 0 for c in last_author_country2paper.keys()}
    for e, d in cite2distance.items():
        if d == 1:  # Collab distance 1 has the bottleneck sample size in the cultural factors plots.
            for c in paper2last_author_country[e[0]]:
                n_collab[c] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    countries_subset = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    print(f"{len(countries_subset)} countries have {thres}+ post-hierarchy citations towards collaborators.")

    # Use country_list.countries_for_language library to get country names from 2-letter codes.
    country_abbr2country_name = {k: {v} for k, v in dict(countries_for_language("en")).items()}

    country_abbr2country_name["US"].add("USA")
    country_abbr2country_name["US"].add("U.S.A.")
    country_abbr2country_name["US"].add("U.S.A")
    country_abbr2country_name["CN"].add("People's Republic of China")
    country_abbr2country_name["CN"].add("PRC")
    country_abbr2country_name["CN"].add("PRoC")
    country_abbr2country_name["GB"].add("UK")
    country_abbr2country_name["GB"].add("U.K.")
    country_abbr2country_name["GB"].add("England")
    country_abbr2country_name["GB"].add("United Kingodm")
    country_abbr2country_name["GB"].add("Great Britain")
    country_abbr2country_name["TR"].add("Türkiye")
    country_abbr2country_name["CZ"].add("Czech Republic")
    country_abbr2country_name["SK"].add("Slovak Republic")
    country_abbr2country_name["KR"].add("Republic of Korea")
    country_abbr2country_name["KR"].add("Korea South")
    country_abbr2country_name["KR"].add("Korea")
    country_abbr2country_name["MX"].add("México")
    country_abbr2country_name["CH"].add("Suisse")

    df_geert = pd.read_csv(os.path.join(dir_6d, "6-dimensions-for-website-2015-08-16.csv"), sep=";", header=0)
    # casefold to tolerate small convention discrepancies
    df_geert["country"] = df_geert["country"].apply(str.casefold)
    # change specially encoded null entries to standard None
    df_geert[df_geert == "#NULL!"] = None

    country2power_distance = {co: None for co in countries_subset}
    country2individualism = {co: None for co in countries_subset}

    for co in countries_subset:
        name_ = country_abbr2country_name[co]
        for na in name_:
            tmp = df_geert[df_geert["country"] == na.casefold()]
            if len(tmp) == 1:
                # tmp[["pdi", "idv", "mas", "uai", "ltowvs", "ivr"]].to_numpy().astype(float)[0]
                pdi, idv = tmp[["pdi", "idv"]].to_numpy().astype(float)[0]
                country2power_distance[co] = pdi
                country2individualism[co] = idv
                break
        else:  # this is run if break condition not met
            print(f"{co} ({name_}) not found")

    country2men_ratio = {co: [0, 0] for co in countries_subset}
    for co in countries_subset:
        for paper in last_author_country2paper[co]:
            last_author = paper2last_author[paper]
            if last_author in last_author2gender_info:
                # Only assign gender to name that has 0.7+ accuracy AND sample size >=20.
                if last_author2gender_info[last_author][1] >= 0.7 and last_author2gender_info[last_author][2] >= 20:
                    country2men_ratio[co][0] += int(last_author2gender_info[last_author][0]=="man")
                    country2men_ratio[co][1] += 1
    
    # Calculate ratios.
    country2men_ratio = {co: val[0]/val[1] for co, val in country2men_ratio.items()}

    savePKL(dir_dict, "country2power_distance", country2power_distance)
    savePKL(dir_dict, "country2individualism", country2individualism)
    savePKL(dir_dict, "country2men_ratio", country2men_ratio)


def save_department_dicts(paper2meta, path_dep, dir_out, print_fail=False):
    df = pd.read_csv(path_dep)
    dep_names = [x.strip().casefold() for x in df["Name"]]
    dep_names_dict = {row: [] for row, x in enumerate(dep_names)}
    for col in df.columns:
        if col != "Name":
            for row, str_ in enumerate(df[col]):
                dep_names_dict[row] += [x.strip().casefold() for x in str_.split(",")]

    paper2last_author_department = {p: set() for p in paper2meta}
    paper2first_author_department = {p: set() for p in paper2meta}
    success, total = 0, 0
    for p in paper2meta:
        aus_dict = paper2meta[p]["authors"]
        if len(aus_dict) == 0:  # No authors found, skip.
            print(f"PMC={p} is skipped because there are no authors found.")
            continue
        affTup = aus_dict[len(aus_dict) - 1]["aff"]  # Last author aff info tuple ("name": last author name; "aff": tuple).
        for aff in affTup[1]:  # affTup[1] is a tuple of aff str
            aff = aff.strip().replace("\n", ",").casefold()
            # Populate department dict.
            found = False  # If we identify any department.
            for row, dep_list in dep_names_dict.items():
                if has_keyword_any(aff, dep_list):
                    paper2last_author_department[p].add(dep_names[row])
                    found = True
            success += int(found)
            total += 1
            if not found and print_fail:
                print(aff)

        affTup = aus_dict[0]["aff"]  # First author aff info tuple ("name": last author name; "aff": tuple).
        for aff in affTup[1]:  # affTup[1] is a tuple of aff str
            aff = aff.strip().replace("\n", ",").casefold()
            found = False  # If we identify any department.
            for row, dep_list in dep_names_dict.items():
                if has_keyword_any(aff, dep_list):
                    paper2first_author_department[p].add(dep_names[row])
                    found = True
            success += int(found)
            total += 1
            if not found and print_fail:
                print(aff)

    fail = total - success
    print(f"success: {success} | fail: {fail} | fail rate {fail / (fail + success)*100:.2f}%")

    paper2last_author_department = {p: list(s) for p, s in paper2last_author_department.items()}
    paper2first_author_department = {p: list(s) for p, s in paper2first_author_department.items()}

    savePKL(dir_out, "paper2last_author_department_28_dep", paper2last_author_department)
    savePKL(dir_out, "paper2first_author_department_28_dep", paper2first_author_department)


def save_department_dicts_all_authors(paper2meta, path_dep, dir_out, print_fail=False):
    df = pd.read_csv(path_dep)
    dep_names = [x.strip().casefold() for x in df["Name"]]
    dep_names_dict = {row: [] for row, x in enumerate(dep_names)}
    for col in df.columns:
        if col != "Name":
            for row, str_ in enumerate(df[col]):
                dep_names_dict[row] += [x.strip().casefold() for x in str_.split(",")]

    paper2all_author_department = {p: [set() for _ in paper2meta[p]["authors"]] for p in paper2meta}
    success, total = 0, 0
    for p in paper2meta:
        aus_dict = paper2meta[p]["authors"]
        if len(aus_dict) == 0:  # No authors found, skip.
            print(f"PMC={p} is skipped because there are no authors found.")
            continue
        for au_order in aus_dict.keys():  # Do it for each author on the author list.
            affTup = aus_dict[au_order]["aff"]  # author aff info tuple ("name": last author name; "aff": tuple).
            for aff in affTup[1]:  # affTup[1] is a tuple of aff str.
                aff = aff.strip().replace("\n", ",").casefold()
                # Populate department dict.
                found = False  # If we identify any department.
                for row, dep_list in dep_names_dict.items():
                    if has_keyword_any(aff, dep_list):
                        paper2all_author_department[p][au_order].add(dep_names[row])
                        found = True
                success += int(found)
                total += 1
                if not found and print_fail:
                    print(aff)

    fail = total - success
    print(f"success: {success} | fail: {fail} | fail rate {fail / (fail + success)*100:.2f}%")

    paper2all_author_department = {p: [list(x) for x in s] for p, s in paper2all_author_department.items()}

    savePKL(dir_out, "paper2all_author_department_28_dep", paper2all_author_department)


def save_benchwork_count(dir_tmp, dir_batch, dir_dict):
    # We don't discard any department.
    benchwork_text_row2response = loadPKL(dir_batch, "benchwork_text_row2response")
    benchwork_text_row2paper = loadPKL(dir_tmp, "benchwork_text_row2paper")
    dep2_100papers = loadPKL(dir_tmp, "dep2_100papers")
    department2benchwork = {dep: [0, 0] for dep in dep2_100papers}
    paper2response = {p: None for p in benchwork_text_row2paper.values()}
    paper_to_discard = set()
    for row, res in benchwork_text_row2response.items():
        paper = benchwork_text_row2paper[row]
        if "yes" in res.casefold():
            r = 1
        elif "no" in res.casefold():
            r = 0
        else:
            print(f"row={row} PMC={paper}, GPT response: {res}, skipped.")
            continue
        if paper2response[paper] is None:
            paper2response[paper] = r
        else:  # Have rated this paper already; if consistent rating, then we don't do anything.
            if paper2response[paper] != r:  # But different rating to the same paper.
                paper_to_discard.add(paper)  # Mark this conflicting paper to remove later.
                print(f"row={row} PMC={paper}, GPT response: ----> {res} <----, different from previous response to the same paper; this paper discarded.")
    for paper in paper_to_discard:
        paper2response.pop(paper)
    for dep, papers in dep2_100papers.items():
        for paper in papers:
            if paper in paper2response:
                department2benchwork[dep][0] += paper2response[paper]
                department2benchwork[dep][1] += 1
    savePKL(dir_dict, "department2benchwork_count", department2benchwork)


def get_department_brilliance(dir_brilliance, departments_subset):
    df_fab = pd.read_csv(os.path.join(dir_brilliance, "brilliance_data.csv"), on_bad_lines="skip", header=0)
    dep2fab_field_label = {d: set() for d in departments_subset}
    ### Below aren't found in the brilliance .csv fields:
    ### "imaging", "mental", "radiology", "behavior", "therapy", "biomedical"
    for x in sorted(list(set(df_fab["field_label"]))):
        x_lowercase = x.casefold()
        if x_lowercase in departments_subset:
            dep2fab_field_label[x_lowercase].add(x)
        elif x in {"MolecularBiology"}:
            dep2fab_field_label["molecular"].add(x)
        elif x in {"AnatomicClinicalPathology"}:
            dep2fab_field_label["pathology"].add(x)
        elif x in {"MedicalGenetics"}:
            dep2fab_field_label["medical"].add(x)
            dep2fab_field_label["genetics"].add(x)
        elif x in {"EmergencyMedicine", "FamilyMedicine", "InternalMedicine", "NuclearMedicine", "PainMedicine"}:
            dep2fab_field_label["medical"].add(x)
        elif x in {"InformationScience"}:
            dep2fab_field_label["information"].add(x)
        elif x in {"Anatomy/CellBiology"}:
            dep2fab_field_label["anatomy"].add(x)
            dep2fab_field_label["cell"].add(x)
        elif x in {"Pharmacology"}:
            dep2fab_field_label["pharma"].add(x)
        elif x in {"ColonRectalSurgery", "GeneralSurgery", "NeurologicalSurgery", "OrthopaedicSurgery", "PlasticSurgery", "ThoracicSurgery"}:
            dep2fab_field_label["surgery"].add(x)
        elif x in {"ComputerScience"}:
            dep2fab_field_label["compute"].add(x)

    # Sample size, mean, standard deviation.
    dep2fab_stats = {d: [np.nan, np.nan, np.nan] for d in departments_subset}

    for d in departments_subset:
        if len(dep2fab_field_label[d]) == 0:  # Skip those not found in brilliance .csv.
            continue
        bl = df_fab["field_label"].isin(dep2fab_field_label[d])
        dep2fab_stats[d][0] = np.sum(bl)  # Each row is a sample, thus this is sample size.
        dep2fab_stats[d][1] = np.mean(df_fab["fab"][bl])  # Brilliance mean.
        dep2fab_stats[d][2] = np.std(df_fab["fab"][bl], ddof=1)  # Brilliance standard deviation. Unused.
    return dep2fab_stats


def save_department_collab_measures(dir_dict, dir_out, thres=100):
    # Get departments_subset.
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    paper2last_author_department = loadPKL(dir_dict, "paper2last_author_department_28_dep")
    # Select subset of departments based on citations.
    last_author_department2paper = reverse_dict_list(paper2last_author_department)
    n_collab = {dep: 0 for dep in last_author_department2paper.keys()}
    for e, d in cite2distance.items():
        if d == 1:  # Collab distance 1 has the bottleneck sample size in the cultural factors plots.
            for dep in paper2last_author_department[e[0]]:
                n_collab[dep] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    departments_subset = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    print(f"{len(departments_subset)} departments have {thres}+ post-hierarchy citations towards collaborators.")

    paper2last_author = loadPKL(dir_dict, "paper2last_author")
    g_coau_t = loadPKL(dir_dict, "g_coau_t")  # Time-varying coauthorship network.
    print("<g_coau_t> loaded.")
    try:
        paper2year = loadPKL(dir_dict, "paper2year")
    except:
        paper2year = loadPKL(dir_dict, "paper2journal_and_year")
        paper2year = {k: v[1] for k, v in paper2year.items()}
    # Only papers in paper2last_author (list), same keys as paper2author_s (set), and paper2meta and thus paper2last_author_department.
    paper2year = {k: v for k, v in paper2year.items() if k in paper2last_author}
    years = []
    for p, year in paper2year.items():
        years.append(year)
    year_min, year_max = min(years), max(years)
    assert (year_min == 1998 and year_max == 2023), f"paper2year range {(year_min, year_max)}, != [1998,2023], change viewxxxx and this if condition accordingly."
    # Only most recent year for author measures, since this is about department authors, not paper level.
    view2023 = nx.subgraph_view(g_coau_t, filter_edge=lambda x, y: g_coau_t.edges[x, y][2023] == 1)
    
    department2last_authors = {dep: {paper2last_author[p] for p in last_author_department2paper[dep]} for dep in departments_subset}
    
    # Avg collab net (most recent year) last author degree.
    department2avg_lau_deg = {dep: np.mean([view2023.degree(au, weight=None) for au in aus]) for dep, aus in tqdm(department2last_authors.items())}

    # Collab net (most recent year) department last authors induced subgraph density.
    department2lau_density = {dep: nx.density(nx.subgraph_view(view2023, filter_node=lambda x: x in aus)) for dep, aus in tqdm(department2last_authors.items())}

    savePKL(dir_out, "department2avg_lau_deg", department2avg_lau_deg)
    savePKL(dir_out, "department2lau_density", department2lau_density)


def save_department_measures(dir_brilliance, dir_dict, thres=50):
    """
    This function involves three thresholds:
        thres (int): Number of post-hierarchy citations towards collaborators.
        Sample size threshold for brilliance and gender: >=20.
        Accuracy threshold for gender: >=70%. 
    """
    paper2meta = loadPKL(dir_dict, "paper2meta")
    paper2last_author_department = loadPKL(dir_dict, "paper2last_author_department_28_dep")
    paper2last_author = loadPKL(dir_dict, "paper2last_author")
    paper2author = loadPKL(dir_dict, "paper2author")  # List ver. of coauthors, not set ver.
    last_author2gender_info = loadPKL(dir_dict, "last_author2gender_info")  # val: ["man"/"woman", accuracy, sample size]
    cite2distance = loadPKL(dir_dict, "cite2distance")  # Collaboration distance.
    department2benchwork = loadPKL(dir_dict, "department2benchwork_count")

    paper2all_author_department_28_dep = loadPKL(dir_dict, "paper2all_author_department_28_dep")
    paper2all_author_country = loadPKL(dir_dict, "paper2all_author_country")

    # Select subset of departments based on citations.
    last_author_department2paper = reverse_dict_list(paper2last_author_department)
    n_collab = {dep: 0 for dep in last_author_department2paper.keys()}
    for e, d in cite2distance.items():
        if d == 1:  # Collab distance 1 has the bottleneck sample size in the cultural factors plots.
            for dep in paper2last_author_department[e[0]]:
                n_collab[dep] += 1
    n_collab = {k: v for k, v in n_collab.items() if v >= thres}
    departments_subset = dict(sorted(n_collab.items(), key=lambda d: d[1], reverse=True))
    print(f"{len(departments_subset)} departments have {thres}+ post-hierarchy citations towards collaborators.")

    # Brilliance measure: len-3 list of sample size, mean, and std of brilliance.
    department2fab_stats = get_department_brilliance(dir_brilliance, departments_subset)
    # Only sample size >=20 is kept, the rest are np.nan.
    department2brilliance = {dep: np.nan for dep in department2fab_stats}
    for dep, stats in department2fab_stats.items():
        if ~np.isnan(stats[0]) and stats[0] >= 20:
            department2brilliance[dep] = stats[1]

    # Remaining department measures are all ratios.
    # 1st in the list is num of review/bench/men, 2nd is total.
    department2benchwork = {dep: department2benchwork[dep] for dep in departments_subset}
    department2synthesis = {dep: [0, 0] for dep in departments_subset}
    department2men_ratio = {dep: [0, 0] for dep in departments_subset}

    department2avg_n_coau = {dep: np.mean([len(paper2author[p]) for p in last_author_department2paper[dep]]) for dep in departments_subset}
    department2n_author = {dep: len({paper2last_author[p] for p in last_author_department2paper[dep]}) for dep in departments_subset}
    department2n_paper = {dep: len(set(last_author_department2paper[dep])) for dep in departments_subset}
    for dep, npap in department2n_paper.items():
        assert len(last_author_department2paper[dep]) == npap, f"{dep} has repeated papers."

    department2avg_n_department = {dep: np.mean([len(set(flatten_list(paper2all_author_department_28_dep[p]))) for p in last_author_department2paper[dep]]) for dep in departments_subset}
    department2avg_n_country = {dep: np.mean([len(set(flatten_list(paper2all_author_country[p]))) for p in last_author_department2paper[dep]]) for dep in departments_subset}

    for dep in departments_subset:
        for paper in last_author_department2paper[dep]:
            t = paper2meta[paper]["article-type"]
            if t == "research-article":
                department2synthesis[dep][1] += 1
            elif t == "review-article":
                department2synthesis[dep][1] += 1
                department2synthesis[dep][0] += 1
            else:
                raise Exception(f"Unknown article type: {t}.")
            last_author = paper2last_author[paper]
            if last_author in last_author2gender_info:
                # Only assign gender to name that has 0.7+ accuracy AND sample size >=20.
                if last_author2gender_info[last_author][1] >= 0.7 and last_author2gender_info[last_author][2] >= 20:
                    department2men_ratio[dep][0] += int(last_author2gender_info[last_author][0]=="man")
                    department2men_ratio[dep][1] += 1
    
    # Calculate ratios.
    department2benchwork = {dep: val[0]/val[1] for dep, val in department2benchwork.items()}
    department2synthesis = {dep: val[0]/val[1] for dep, val in department2synthesis.items()}
    department2men_ratio = {dep: val[0]/val[1] for dep, val in department2men_ratio.items()}

    savePKL(dir_dict, "department2synthesis", department2synthesis)
    savePKL(dir_dict, "department2benchwork", department2benchwork)
    savePKL(dir_dict, "department2men_ratio", department2men_ratio)
    savePKL(dir_dict, "department2brilliance", department2brilliance)
    savePKL(dir_dict, "department2avg_n_coau", department2avg_n_coau)
    savePKL(dir_dict, "department2n_author", department2n_author)
    savePKL(dir_dict, "department2n_paper", department2n_paper)
    savePKL(dir_dict, "department2avg_n_department", department2avg_n_department)
    savePKL(dir_dict, "department2avg_n_country", department2avg_n_country)

