import os
import csv
import tarfile
import warnings
import tqdm
# import urllib.request
import requests
import random
import json


def get_ftp_path(csv_in):
    """
    process oa_file_list_filtered.csv file in <csv_in> (e.g., /data),
    output a list of its first column (i.e., relative ftp path)
    """
    kwargs = dict(encoding="UTF-8", newline="")
    ftp_path_dict = dict()
    for file in os.listdir(csv_in):
        if file.endswith(".csv"):
            with open(os.path.join(csv_in, file), mode="r", **kwargs) as file_in:
                spamreader = csv.reader(file_in)
                next(spamreader)  # skip header
                for row in spamreader:
                    pmcid = os.path.basename(row[0]).split(".")[0][3:]
                    ftp_path_dict[pmcid] = row[0]  # first col, which is ftp path
                    # e.g., "oa_package/72/19/PMC1305130.tar.gz"
                    if not row[0].endswith(".tar.gz"):
                        raise Exception(f"ftp path {row[0]} in csv {file} is not a tgz file")

    return ftp_path_dict


def extract_tgz2(tgz_path, content_out, xml_records):
    """this is NOT a general purpose extraction function
    this one uses 2nd gen xml_records (which uses pmcid (number only, but in str format) as the key)
    if there are 2+ xml files, pick the one that is .nxml first, and also pick the first one
    Args
    ----
    - tgz_path (str): path of the tgz file we have locally
    - content_out (str): dir (folder) of the extracted files
    """
    pmcid = os.path.basename(tgz_path).split(".")[0][3:]
    try:
        with tarfile.open(tgz_path, "r:gz") as f:  # unpack the tgz file
            list_of_names = f.getnames()
            xml_path = [x for x in list_of_names if x.endswith("xml")]
            num_xml = len(xml_path)
            if num_xml == 1:
                xml_records[f"{pmcid}"]["issues"] = f"none"
            elif num_xml > 1:
                warnings.warn(f"{pmcid} (pmcid) has {num_xml} xml files")
                xml_path = [x for x in xml_path if x.endswith("nxml")]  # e.g., PMC6215330
                xml_records[f"{pmcid}"]["issues"] = f"1. {num_xml} xml files (only '{xml_path[0]}' saved)"
            else:  # num_xml=0
                warnings.warn(f"{pmcid} (pmcid) has no xml files")
                xml_records[f"{pmcid}"]["issues"] = f"2. no xml files"
                return 0  # stop here since there's nothing we want to extract
            # if >1, only save the first one
            xml_path = xml_path[0]  # e.g., "PMC1305130/1283.nxml"; relative path
            f.extractall(content_out, members=[f.getmember(xml_path)])
            xml_path = os.path.join(content_out, xml_path)  # absolute path
            xml_path = os.path.normpath(xml_path)  # fix slashes
            # e.g., xml_path = f"{content_out}\PMC1305130\1283.nxml"
    except Exception as err:
        print("An exception occurred:", err)
        warnings.warn(f"PMC{pmcid} has no xml files we can obtain because of the error")
        xml_records[f"{pmcid}"]["issues"] = f"3. no xml files possible because of an error"
        os.remove(tgz_path)  # remove the tgz file (that is not tgz file)
        print(f"PMC{pmcid} is NOT downloaded. ({tgz_path})")
        return 0
    # clean up the files
    os.rename(xml_path, os.path.join(content_out, f"{pmcid}.xml"))  # move file outside of dir
    os.rmdir(os.path.dirname(xml_path))  # delete said dir
    os.remove(tgz_path)  # remove the tgz file since we unpacked the desired content


def download_tgz_files(csv_in, xml_out):
    """
    if xml file in the tgz file has an extension of "nxml", repace it with "xml"

    last time I checked:
    oa_package/f4/b8/PMC6090298.tar.gz,Front Neurosci. 2018 Aug 7; 12:529,PMC6090298,2019-12-06 19:17:31,30131669,CC BY
    doesn't exist when I went to the ftp server on browser,
    so there might be an error about it not being a gzip file

    output
    ------
    - xml_records (nested dict): key is pmcid in ftp_path_dict (only numbers, no "PMC" letters);
        if we have the key, we tried to download it (check "issues" to see the results)
        val is a dict:
        - "ftp_path": ftp_path of ftp_path_dict
        - "PMCID": pmcid in ftp_path of ftp_path_dict
        - "issues": descriptive string; empty if no issues
    """
    temp = os.listdir(xml_out)

    xml_records_name = f"xml_records.json"
    xml_records_path = os.path.join(xml_out, xml_records_name)
    if xml_records_name in temp:  # if xml_records json file exists
        with open(xml_records_path, mode="r") as f:  # read it
            xml_records = json.load(f)
    else:  # if it doesn't (meaning no xml downloaded)
        xml_records = dict()  # then make it
    # either way we will download and then save it (overwrite if it exists)

    ftp_path_dict = get_ftp_path(csv_in)

    # list_url = []
    # list_file_path = []

    url_prefix = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
    try:
        for idx, ftp_path in tqdm.tqdm(ftp_path_dict.items()):
            if f"{idx}" not in xml_records:  # if False, then we've tried to download it
                fname_full = os.path.basename(ftp_path)  # e.g., "PMC1305130.tar.gz"
                pmcid = fname_full.split(".", maxsplit=1)[0]  # only need name before first dot; "PMC1305130"
                xml_records[f"{idx}"] = {"ftp_path": ftp_path, "pmcid": pmcid}
                url = url_prefix + ftp_path
                tgz_path = os.path.join(xml_out, fname_full)
                # urllib.request.urlretrieve(url, tgz_path)
                with requests.get(url, stream=True, allow_redirects=True) as resp:
                    with open(tgz_path, "wb") as f:  # save the tgz file
                        f.write(resp.raw.read())
                # list_url.append(url)
                # list_file_path.append(tgz_path)
                extract_tgz2(tgz_path, xml_out, xml_records)  # unpack the tgz file
                # print(f"#{idx} downloaded. (fname_full={fname_full})")
            # save xml_records as json file once in a while
            if random.random() < 0.05:
                with open(xml_records_path, "w") as f:
                    json.dump(xml_records, f, indent=4)
    except Exception as err:  # in case unexpected error happens, we save the records first
        with open(xml_records_path, "w") as f:
            json.dump(xml_records, f, indent=4)
        print(f"DEBUG current idx={idx}; pmcid={pmcid}")
        print(f"DEBUG Unexpected {err=}, {type(err)=}")
        raise
