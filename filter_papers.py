import os
import pickle
import csv
import re
import pandas as pd


def save_journal_meta(dir_in, dir_out):
    df = pd.read_csv(os.path.join(dir_in, "JCR_JournalResults_05_2024.csv"), on_bad_lines="skip", header=2)
    df = df.drop([305, 306, 307, 308])  # 305 is one whose 2022 JIF is "<0.1"
    topJournals_tits = []
    topJournals_abbr = []
    topJournals_issn = []
    topJournals_essn = []
    issn2data = dict()
    essn2data = dict()

    for _, row in df[(pd.to_numeric(df["2022 JIF"]) >= 3)].iterrows():
        topJournals_tits.append(row["Journal name"].casefold())
        topJournals_abbr.append(row["JCR Abbreviation"].casefold())
        td = dict()
        td["tot_cite"] = pd.to_numeric(row["Total Citations"].replace(",", ""))  # 2022
        td["JIF2022"] = pd.to_numeric(row["2022 JIF"])
        td["OAGoldPercent"] = pd.to_numeric(row["% of OA Gold"].replace("%", ""))  # 2022; AKA "% OF CITABLE OA"
        if pd.isna(row["ISSN"]):
            topJournals_issn.append(None)
        else:
            topJournals_issn.append(row["ISSN"])
            issn2data[row["ISSN"]] = td
        if pd.isna(row["eISSN"]):
            topJournals_essn.append(None)
        else:
            topJournals_essn.append(row["eISSN"])
            essn2data[row["eISSN"]] = td

    topJournals_tits_s = set(topJournals_tits)
    topJournals_abbr_s = set(topJournals_abbr)
    topJournals_issn_s = set(topJournals_issn)
    topJournals_essn_s = set(topJournals_essn)

    # Manually add 3 relevant journals outside of WoS-neurosciences journals.
    topJournals_issn_s.add("0013-9580")
    topJournals_essn_s.add("1528-1167")
    td = {"tot_cite": 32302, "JIF2022": 5.6, "OAGoldPercent": 27.45}
    issn2data["0013-9580"] = td
    essn2data["1528-1167"] = td

    topJournals_issn_s.add("2213-1582")
    topJournals_essn_s.add("2213-1582")
    td = {"tot_cite": 14668, "JIF2022": 4.2, "OAGoldPercent": 99.65}
    issn2data["2213-1582"] = td
    essn2data["2213-1582"] = td

    topJournals_issn_s.add("1759-4758")
    topJournals_essn_s.add("1759-4766")
    td = {"tot_cite": 19735, "JIF2022": 38.1, "OAGoldPercent": 4.17}
    issn2data["1759-4758"] = td
    essn2data["1759-4766"] = td

    for s in [topJournals_tits_s, topJournals_abbr_s, topJournals_issn_s, topJournals_essn_s]:
        if None in s:
            s.remove(None)

    substr_ = ["SEP:", "JrId:", "JournalTitle:", "MedAbbr:", "ISSN (Print):", "ISSN (Online):", "IsoAbbr:", "NlmId:"]
    # 2 methods to match:
    # "ISSN (Print)" matches "ISSN", "ISSN (Online)" matches "eISSN"
    founD = dict()
    with open(os.path.join(dir_in, "J_Medline.txt"), "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            idx, e = divmod(i, 8)
            if e == 4:  # ISSN (Print)
                tmp = line.replace(substr_[e], "").strip()
                if tmp in topJournals_issn_s:
                    founD[idx] = dict()
            elif e == 5:  # ISSN (Online)
                tmp = line.replace(substr_[e], "").strip()
                if tmp in topJournals_essn_s:
                    founD[idx] = dict()

    # Fill founD that's already populated with keys (but no values yet).
    with open(os.path.join(dir_in, "J_Medline.txt"), "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            idx, e = divmod(i, 8)
            if idx in founD and e != 0:
                founD[idx][substr_[e][:-1]] = line.replace(substr_[e], "").strip()

    for k in founD:
        a, b = founD[k]["ISSN (Print)"], founD[k]["ISSN (Online)"]
        if a or b:
            if a in issn2data:
                founD[k]["jourMeta"] = issn2data[a]
            elif b in essn2data:
                founD[k]["jourMeta"] = essn2data[b]
            else:
                raise Exception("Neither Print nor Online ISSN is found.")

    with open(os.path.join(dir_out, "jour2meta.pkl"), "wb") as f:
        pickle.dump(founD, f)


def make_year_range_list():
    # reason for 1998 see:
    # [2018] Characterizing in-text citations in scientific articles: A large-scale analysis
    return [1998, 2023]


def make_journal_list():
    """the str in the list is "MedAbbr" in J_Medline.txt, which contains all journals PMC has
    (total 188, since 1 PMC didn't have, it's 187)
    neuroscience:
    WoS neurosciences journals (JIF>=3) + 3 more neuroscience journals:
        Epilepsia                   2022 JIF = 5.6  (obtained from WoS)
        NeuroImage: Clinical        2022 JIF = 4.2  (obtained from WoS)
        Nature Reviews Neurology    2022 JIF = 38.1 (obtained from WoS)
    """
    # some top ones
    journal_list = ["Nat Neurosci", "Neuron", "Brain", "J Neurosci", "Neuroimage"]
    journal_list += ["Nat Rev Neurosci", "Cereb Cortex", "Ann Neurol", "Hum Brain Mapp", "Epilepsia"]
    journal_list += ["Clin Neurophysiol", "Neurosci Biobehav Rev", "Neuropsychopharmacology"]
    journal_list += ["Neuroimage Clin", "Neurobiol Aging", "Trends Neurosci", "Nat Rev Neurol"]
    journal_list += ["Brain Stimul", "Front Neurosci", "Front Neurol", "Cortex"]

    # JIF >=3.5
    journal_list += ['Behav Brain Sci', 'Mol Cell Neurosci', 'J Neurodev Disord', 'Acta Neuropathol Commun', 'Mol Neurobiol', 'Acta Neuropathol', 'Mol Brain', 'Behav Brain Funct', 'Rev Neurosci', 'ACS Chem Neurosci', 'J Physiol', 'Neurochem Int', 'J Neurol Sci', 'Int J Tryptophan Res', 'Neural Regen Res', 'J Neural Eng', 'Psychoneuroendocrinology', 'Neurosci Bull', 'Cell Mol Neurobiol', 'Curr Opin Behav Sci', 'Front Aging Neurosci', 'Neurobiol Dis', 'Front Neuroendocrinol', 'Curr Neuropharmacol', 'Dialogues Clin Neurosci', 'J Peripher Nerv Syst', 'Neuroscientist', 'Eur Neuropsychopharmacol', 'Int J Neuropsychopharmacol', 'Cogn Neurodyn', 'Neural Netw', 'Neuropsychol Rev', 'Pharmacol Biochem Behav', 'Metab Brain Dis', 'Neurosci Insights', 'IEEE Trans Cogn Dev Syst', 'J Psychopharmacol', 'Nat Aging', 'Front Mol Neurosci', 'Cognit Comput', 'Brain Res Bull', 'Glia', 'Cephalalgia', 'Bipolar Disord', 'Neurophotonics', 'Front Neuroinform', 'J Cereb Blood Flow Metab', 'Sleep', 'Psychophysiology', 'J Parkinsons Dis', 'J Alzheimers Dis', 'Neural Dev', 'Mol Autism', 'Brain Behav Immun', 'J Neuroinflammation', 'Ann Clin Transl Neurol', 'Front Integr Neurosci', 'Neurobiol Stress', 'Transl Stroke Res', 'Neurol Neuroimmunol Neuroinflamm', 'Curr Neurol Neurosci Rep', 'J Pain', 'Network', 'Brain Pathol', 'Psychiatry Clin Neurosci', 'Front Neural Circuits', 'Neurogastroenterol Motil', 'Mult Scler', 'Sleep Med Rev', 'Eur J Neurol', 'Annu Rev Vis Sci', 'NPJ Parkinsons Dis', 'Curr Opin Neurol', 'NPJ Sci Learn', 'Nutr Neurosci', 'Neurochem Res', 'J Pineal Res', 'J Neurotrauma', 'Front Synaptic Neurosci', 'Dev Cogn Neurosci', 'J Neuroeng Rehabil', 'Annu Rev Neurosci', 'J Neurosci Res', 'Chem Senses', 'Neuroendocrinology', 'Fluids Barriers CNS', 'Acta Neuropsychiatr', 'Alzheimers Dement (N Y)', 'Clin Auton Res', 'Cogn Syst Res', 'Prog Neuropsychopharmacol Biol Psychiatry', 'Purinergic Signal', 'Exp Neurol', 'Front Cell Neurosci', 'Brain Commun', 'J Psychiatry Neurosci', 'Curr Opin Neurobiol', 'J Neurochem', 'J Neuroimmune Pharmacol', 'Prog Neurobiol', 'Transl Neurodegener', 'J Sleep Res', 'Trends Cogn Sci', 'Alzheimers Res Ther', 'Biol Psychiatry Cogn Neurosci Neuroimaging', 'Cerebellum', 'J Headache Pain', 'ASN Neuro', 'Neuropathol Appl Neurobiol', 'Soc Cogn Affect Neurosci', 'Alzheimers Dement (Amst)', 'Neurotox Res', 'Hippocampus', 'Biol Psychiatry', 'Mol Psychiatry', 'Mol Neurodegener', 'Neuromolecular Med', 'Netw Neurosci', 'Pain Rep', 'Eur J Pain', 'Neurotherapeutics', 'Nat Hum Behav', 'Pain', 'Neuropharmacology', 'CNS Neurosci Ther']

    # 3<= JIF <3.5
    journal_list += ['J Cogn Neurosci', 'Front Neurorobot', 'eNeuro', 'J Neurovirol', 'J Mol Neurosci', 'J Neuropathol Exp Neurol', 'Neural Plast', 'Dev Neurobiol', 'Seizure', 'Neuroinformatics', 'Brain Behav', 'Neurotoxicology', 'Front Syst Neurosci', 'CNS Neurol Disord Drug Targets', 'Mol Pain', 'Neurodegener Dis', 'Muscle Nerve', 'Front Comput Neurosci', 'Trends Neurosci Educ', 'Brain Struct Funct', 'Neurophysiol Clin', 'Brain Sci', 'Neurobiol Lang (Camb)', 'Neuroscience', 'Neurol Sci', 'Epilepsia Open', 'Nat Sci Sleep', 'Neuropsychobiology', 'J Neural Transm (Vienna)', 'Int J Psychophysiol', 'Brain Connect', 'J Neuroimmunol', 'Eur J Neurosci', 'Clocks Sleep', 'J Huntingtons Dis', 'J Neuromuscul Dis', 'J Neuroendocrinol', 'Psychopharmacology (Berl)', 'Clin Psychopharmacol Neurosci', 'J Neurosci Methods', 'J Alzheimers Dis Rep']

    return journal_list


def init_stats_dict():
    """
    stats_dict (dict):
        "year_range": [min,max] of all entries
            with a year (but not necessarily a journal) in "Article Citation"
        "year_range_j": [min,max] of all entries
            with a year and a journal in "Article Citation"
        "journal_year_lookup" (dict): key is PMC (str); val is [journal, year]
            depending on the kwargs in:
            _csv_filter(row, stats_dict, year_range=None, journal_set=None)
            if both year is not None, AND journal is not None, then this dict is
            only for papers with both a year and a journal in "Article Citation" in "oa_file_list.csv"

        all keys except "journal_year_lookup" and "year_range" have int for num of papers
    """
    stats_dict = dict(included=0, filtered=0, no_year=0, no_journal=0, other=0, total=0)
    stats_dict["no_journal_no_year"] = 0
    stats_dict["year_range"] = [9999, -9999]
    stats_dict["year_range_j"] = [9999, -9999]
    stats_dict["journal_year_lookup"] = dict()
    return stats_dict


def filter_file_list(year_range, journal_set, dir_in=None, dir_out=None):
    """
    process oa_file_list.csv, which is a table of basic info of ALL files available on PMC OA

    save 3 files in /data:
        filtered oa_file_list
        stats file of the original oa_file_list
        stats file of the filtered oa_file_list

    the filelist file should be located in /input
        'tis one big (~700 MB) file having both comm or noncomm entries

    Kwargs
    ------
    - dir_in: directory containing "oa_file_list.csv"
    - dir_out: directory to save to:
        filtered "oa_file_list_filtered.csv"
        "stats_dict.pkl"
        "stats_dict_filtered.pkl"

    """
    fname_csv = "oa_file_list.csv"
    fname_csv_filtered = "oa_file_list_filtered.csv"
    fname_pkl = "stats_dict.pkl"
    fname_pkl_filtered = "stats_dict_filtered.pkl"

    fname_csv = os.path.join(dir_in, fname_csv)
    fname_csv_filtered = os.path.join(dir_out, fname_csv_filtered)

    # read and filter <fname_csv> & save <fname_csv_filtered>
    # filters happening here
    stats_dict, stats_dict_filtered = _filelist_filter(fname_csv, fname_csv_filtered, year_range, journal_set)
    # save both dict files
    with open(os.path.join(dir_out, fname_pkl), "wb") as f:
        pickle.dump(stats_dict, f)
    with open(os.path.join(dir_out, fname_pkl_filtered), "wb") as f:
        pickle.dump(stats_dict_filtered, f)

    return stats_dict, stats_dict_filtered


def _filelist_filter(path_in, path_out, year_range, journal_set):
    """
    filter oa_file_list.csv based on year and journals

    Args
    ----
    - path_in: oa_file_list.csv path to read and filter
    - path_out: oa_file_list_filtered.csv to write to

    Return
    ------
    - stats_dict_filtered (dict):
        "journal_year_lookup" (dict): key is PMC (str); val is [journal, year]
        (the remaining values are all integers)
        "included":
        "filtered": outside of journal set or year range
        "no_year":
        "no_journal":
        "other": whatever's left that does not fit the above
        above ones are mutually exclusive
        "total": count of rows in filelist files
    - stats_dict (dict): ditto except it's unfiltered

    """

    stats_dict = init_stats_dict()
    stats_dict_filtered = init_stats_dict()
    kwargs = dict(encoding="UTF-8", newline="")
    with open(path_out, mode="w", **kwargs) as file_out:
        spamwriter = csv.writer(file_out)
        with open(path_in, mode="r", **kwargs) as file_in:
            spamreader = csv.reader(file_in)
            spamwriter.writerow(next(spamreader))  # read and write header
            for row in spamreader:
                _csv_filter(row, stats_dict, year_range=None, journal_set=None)
                if _csv_filter(row, stats_dict_filtered, year_range, journal_set):
                    spamwriter.writerow(row)

    return stats_dict, stats_dict_filtered


def _csv_filter(row, stats_dict, year_range=None, journal_set=None):
    """
    filter the entries/rows in oa_file_list.csv
        based on journals and publication years
    filters such as EN language need xml files instead, so they aren't done here

    Args
    ----
    if year_range/journal_set is None, then we don't use that as a filter
    - year_range (list; len=2):
        for "comm" subset, earliest year is 2003
        for all OA subset, year range is [1781, 2024]
    - journal_set (list):
        journals to include

    fields
    'Article File', 'Article Citation', 'AccessionID',
    'LastUpdated (YYYY-MM-DD HH:MM:SS)', 'PMID', 'License', 'Retracted'

    "Article Citation" is a string of journal abbreviation and date
        journal abbreviation always ends with a period
        if the period is followed by space, then it has publication year
        (publication year always starts with a space, might be padded with a 0 making it 5-digit)
        if the period is followed by semicolon, then it doesn't have publication year
        there should not be any other symbol that immediately follows the period
    """
    stats_dict["total"] += 1
    # only need journal and year, so group only these two, but temp.group(0) is the full cell
    temp = re.search(r"(.+)\. ([0-9]{4})[ ;].*", row[1])
    if temp:  # has journal has year
        journal = temp.group(1)
        year = int(temp.group(2))
        pmcid = int(row[2][3:])
        if year < stats_dict["year_range"][0]:
            stats_dict["year_range"][0] = year
        if year > stats_dict["year_range"][1]:
            stats_dict["year_range"][1] = year

        if year < stats_dict["year_range_j"][0]:
            stats_dict["year_range_j"][0] = year
        if year > stats_dict["year_range_j"][1]:
            stats_dict["year_range_j"][1] = year

        if ((journal_set is None) or (journal in journal_set)) and ((year_range is None) or (year_range[0] <= year <= year_range[1])):
            stats_dict["included"] += 1
            stats_dict["journal_year_lookup"][pmcid] = [journal, year]
            return True
        else:
            stats_dict["filtered"] += 1
            return False
    elif re.search(r"(.+)\.(?! ).+", row[1]):  # no year but has journal
        stats_dict["no_year"] += 1
        return False
    elif re.search(r"^ ([0-9]{4})[ ;].*", row[1]):  # no journal but has year
        stats_dict["no_journal"] += 1
        year = int(re.search(r"^ ([0-9]{4})[ ;].*", row[1]).group(1))
        if year < stats_dict["year_range"][0]:
            stats_dict["year_range"][0] = year
        if year > stats_dict["year_range"][1]:
            stats_dict["year_range"][1] = year
        return False
    elif re.search(r"^;.*", row[1]):  # no journal no year
        stats_dict["no_journal_no_year"] += 1
        return False
    else:
        print("below row's 2nd col doesn't follow expected formatting behavior:")
        print(row[0:2])
        stats_dict["other"] += 1
        return False
