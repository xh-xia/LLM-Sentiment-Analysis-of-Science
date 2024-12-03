import os
import pickle
import json
import openai
import tiktoken
import time
import requests  # For WOS.
import traceback  # For WOS.
import numbers  # For WOS.
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

# import metapub as mp # time.sleep(0.5) cuz <=3 requests/sec; otherwise need API key for fetch (from NCBI).
from collections import defaultdict

from pubmed_parser import parse_pubmed_paragraph
from helper_functions import reverse_dict_val, reverse_dict_list

ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")
# Context window for GPT-3.5-turbo is 16,385 tokens.
MAX_TOKEN = 10000  # Intro and some results/methods to get a good idea on whether a paper is benchwork. 10K is more than enough.


def find_nth_overlapping(haystack, needle, n):
    """
    n starts from 1 (which means 1st, and 2=2nd, and so on)
    -1 if find() can't find it
    """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + 1)
        n -= 1
    return start


def make_subset_from_cnets(cnets):
    """
    Return
    ------
    - largest weakly connected components (from cnets["edges"])

    """
    G = nx.from_edgelist(cnets["edges"], create_using=nx.DiGraph)
    tmp = list(nx.weakly_connected_components(G))
    tmp = sorted(tmp, key=lambda x: len(x), reverse=True)  # list of sets (connected component)
    sizes = [len(x) for x in tmp]  # list of connected component sizes
    print(f"weakly connected components stats:")
    print(f"total: {sum(sizes)} articles")
    print(f"largest component ('subset'): {sizes[0]} articles")
    print(f"remaining {len(sizes[1:])} components:\n\t{sum(sizes[1:])} articles")
    print(f"\tmean: {np.mean(sizes[1:]):.2f} articles")
    print(f"\tstd: {np.std(sizes[1:]):.2f} articles")
    print(f"\tmedian: {np.median(sizes[1:])} articles")

    return tmp[0]


def save_CGPT_input_files(dir_cnets, dir_out, subset=None, cite_marker="✪"):
    """generate input .txt files for ChatGPT (v3.5) ("CGPT")
    remove "()" & "[]" for all sentences involved here

    Kwarg
    -----
    - subset (list-like, preferably set):
        subset of articles to generate sentences from
        default: generated by make_subset_from_cnets()

    Intermediaries
    --------------
    sample sentences from the in-network citation sentences uniformly at random, without replacement
    sent_idx2netidx_lookup:
        key is sentence idx; val is paper's in-network idx
    new_marker: citation marker for the current in-network in-text citation
    each time an in-network in-text citation shows up in a sentence, we add a row
    so if a sentence has 2 chains of in-network in-text citation, we add 2 rows
    also, we remove all non-current cite_marker, in-network or not, so that each row only contains one citation

    Save
    ----
    - "sentences2rate-CGPT.txt"
        each line contains a sentence for CGPT to rate
    - sentrow2edgeinfo (dict): saved as a pkl file
        info for those sentences to be sent to CGPT
        key: row_idx in the txt file
        val (list):
            val[0]: "sents" sentence idx
            val[1]: (int): citing paper idx; pmcid
            val[2]: (list of int): cited papers idxs; pmcid
            val[3]: (int or None) rating to be had

    """
    with open(os.path.join(dir_cnets, "cnets2_Neuroscience.pkl"), "rb") as f:
        cnets = pickle.load(f)
    # find subset of articles to pick sentences from
    if subset is None:
        subset = make_subset_from_cnets(cnets)
    # i here is "sents" sentence idx
    all_sents = {i: j for v in cnets["sents"].values() for (i, j) in v.items()}
    all_cites = {i: j for v in cnets["cites"].values() for (i, j) in v.items()}
    sent_idx2pmcid_lookup = {k: i for i in cnets["sents"] for k in cnets["sents"][i]}

    sentrow2edgeinfo = dict()
    samp_idx = all_sents.keys()  # sentence idx

    kwargs = dict(encoding="UTF-8")
    with open(os.path.join(dir_out, "sentences2rate-CGPT.txt"), mode="w+", **kwargs) as file_out:
        row_idx = 0
        for idx in samp_idx:
            if sent_idx2pmcid_lookup[idx] not in subset:  # skip ones not in subset
                continue
            for i, c in enumerate(all_cites[idx]):  # c is a set (for each citation chain)
                c.discard(None)
                if len(c) >= 1:  # has at least one in-network citation
                    # replace the current cite_marker with the new_marker (and then return it)
                    # this way we can remove all OTHER cite_marker
                    tmp_idx = find_nth_overlapping(all_sents[idx], cite_marker, i + 1)
                    new_marker = "TISATEMPERORYPLACEHOLDER"
                    tmp_sent = f"{all_sents[idx][:tmp_idx]}{new_marker}{all_sents[idx][tmp_idx+1:]}"
                    tmp_sent = tmp_sent.replace(cite_marker, "")  # remove all other cite_marker
                    tmp_sent = tmp_sent.replace(new_marker, cite_marker)  # return the cite_marker
                    tmp_sent = tmp_sent.replace("()", "")
                    tmp_sent = tmp_sent.replace("[]", "")
                    file_out.write(tmp_sent + "\n")
                    sentrow2edgeinfo[row_idx] = [idx, sent_idx2pmcid_lookup[idx], [s for s in c], None]
                    row_idx += 1

    with open(os.path.join(dir_out, "sentrow2edgeinfo.pkl"), "wb") as f:
        pickle.dump(sentrow2edgeinfo, f)


def CGPT_init(api_key):
    client = openai.OpenAI(api_key=api_key)
    system_prompt = """For each in-text citation, the rater should measure the sentiment of the citing research toward the cited research (represented as the character ✪), on a scale of -1 to 1. 
    The rater should assign a positive score (+1) to statements depicting the cited research as positive, corroborative, consistent with, similar to, or in common with the citing research.
    Conversely, the rater should assign a negative score (-1) to statements depicting the cited research as negative, refuting, inconsistent with, dissimilar to, or different from the citing research.
    If the statements are neutral or do not belong to the aforementioned categories, then the rater should assign 0 to the statements. 
    When you are given a sentence only answer with the numerical results without explanation."""
    user_prompt = "The sentence to analyze is: "
    init_dict = {"client": client, "system_prompt": system_prompt, "user_prompt": user_prompt}
    return init_dict


def CGPT_init_benchwork(api_key):
    client = openai.OpenAI(api_key=api_key)
    system_prompt = 'You are a reviewer with expertise in assessing the data sources of research papers. Your task is to determine whether a paper is based on data collected directly by the authors, or from data from other sources, such as using data from another lab, previous research, or relying on computational or theoretical methods. Answer only "Yes" if the authors collected their own data, or only "No" if the data came from external sources, simulations, or purely theoretical origins.'
    user_prompt = "The paper snippet is: "
    init_dict = {"client": client, "system_prompt": system_prompt, "user_prompt": user_prompt}
    return init_dict


def get_txt_from_paragraphs(paragraphs):
    current_token__count = 0
    txt_to_sent = ""
    for i, pa in enumerate(paragraphs):
        txt = pa["text"].replace("\n", "").strip()  # Remove newlines and strip.
        txt_enc = ENC.encode(txt)
        if current_token__count + len(txt_enc) > MAX_TOKEN:
            break
        txt_to_sent += txt
        current_token__count += len(txt_enc)
    if len(txt_to_sent) == 0:
        raise Exception(f"PMC={paragraphs[0]['pmc']} first section already goes over token limits.")
    return txt_to_sent


def save_paper_snippet(dir_xml, dir_dict, dir_tmp, n_paper=100):
    """randomly sample n_paper research (not review) papers from each department.

    Args:
        dir_xml: Folder holding xml files.
        dir_dict: Folder holding paper2meta and paper2last_author_department_28_dep.
        dir_tmp: Folder holding ChatGPT related files for downstream processing.
        n_paper (int): Num of papers randomly drawn per department, without replacement.
    """
    with open(os.path.join(dir_dict, "paper2meta.pkl"), "rb") as f:
        paper2meta = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2last_author_department_28_dep.pkl"), "rb") as f:
        paper2last_author_department = pickle.load(f)  # Last author departments for each paper.
    last_author_department2paper = reverse_dict_list(paper2last_author_department)
    # Only research papers.
    last_author_department2paper = {
        d: [p for p in ps if paper2meta[p]["article-type"] == "research-article"] for d, ps in last_author_department2paper.items()
    }
    for d, vs in last_author_department2paper.items():
        if len(vs) < n_paper:
            raise Exception(f"{d} has {len(vs)} research papers, less than {n_paper}.")
    rng = np.random.default_rng()
    kwargs = dict(encoding="UTF-8")
    dep2_papers = {d: rng.choice(ps, size=len(ps), replace=False) for d, ps in last_author_department2paper.items()}
    dep2_100papers = {d: [] for d in last_author_department2paper}
    files = set(file for file in os.listdir(dir_xml) if file.endswith(".xml"))
    benchwork_text_row2paper = dict()
    with open(os.path.join(dir_tmp, "benchwork_text_CGPT.txt"), mode="w+", **kwargs) as file_out:
        c = 0
        for d, ps in dep2_papers.items():
            n_found = 0  # We need n_paper papers for each department.
            for p in ps:  # Get snippet of each paper.
                tmp1 = f"{p}.xml"
                if tmp1 not in files:
                    tmp1 = f"PMC{p}.xml"
                if tmp1 not in files:
                    raise Exception(f"department {d}, paper PMC={p} is not in {dir_xml}.")
                tmp1 = os.path.join(dir_xml, tmp1)
                para = parse_pubmed_paragraph(tmp1, all_paragraph=True)  # We don't care about ref here, so True.
                if len(para) == 0:
                    print(f"department {d} PMC={p} has no paragraphs found.")
                    continue  # Skip this paper.
                line = get_txt_from_paragraphs(para)
                file_out.write(line + "\n")
                dep2_100papers[d].append(p)
                benchwork_text_row2paper[c] = p
                n_found += 1
                c += 1
                if n_found == n_paper:
                    break  # Next department.
            if n_found != n_paper:
                raise Exception(f"After parsing, not enough papers in {d}.")

    with open(os.path.join(dir_tmp, "dep2_100papers.pkl"), "wb") as f:
        pickle.dump(dep2_100papers, f)
    with open(os.path.join(dir_tmp, "benchwork_text_row2paper.pkl"), "wb") as f:
        pickle.dump(benchwork_text_row2paper, f)

    with open(os.path.join(dir_tmp, "benchwork_text_CGPT.txt"), mode="r+", **kwargs) as file_out:
        out = file_out.readlines()
    n1 = len(benchwork_text_row2paper)
    n2 = int(len(dep2_100papers) * n_paper)
    if len(out) != n1 or len(out) != n2 or n1 != n2:
        raise Exception(f"Please rerun this function.")


def get_rating(citation_text, init_dict, model=None):
    if model is None:
        model = "gpt-3.5-turbo-0125"  # 2024 model untrained, not as good as 2023 one for sentiment; but better for benchwork detection
        model = "gpt-3.5-turbo-1106"  # 2023 model untrained; use a trained model if possible, but otherwise use this one
    messages = [{"role": "system", "content": init_dict["system_prompt"]}]
    messages.append({"role": "user", "content": init_dict["user_prompt"] + citation_text})
    try:
        chat_completion = init_dict["client"].chat.completions.create(model=model, temperature=0.01, messages=messages)
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)
        return None
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
        print(e)
        return None
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
        return None
    response = chat_completion.choices[0].message.content

    return response


def get_embedding(client, text, model=None):
    if model is None:
        model = "text-embedding-3-small"
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_title_embedding(dir_dict, dir_temp, api_key, model=None):
    client = openai.OpenAI(api_key=api_key)
    path_embed = os.path.join(dir_temp, "title_embedding.pkl")
    # Get titles parsed from xml.
    with open(os.path.join(dir_dict, "paper2full_title.pkl"), "rb") as f:
        paper2full_title = pickle.load(f)
    # Get (partially) saved data.
    if os.path.exists(path_embed):
        with open(path_embed, "rb") as f:
            saved_data = pickle.load(f)
    else:
        saved_data = dict()

    new_done = 0
    for i, (p, title_str) in tqdm(enumerate(paper2full_title.items())):
        if p not in saved_data:
            time.sleep(0.023)  # Assuming Tier-1 in OpenAI, then 0.02 is rate limit for RPM.
            saved_data[p] = get_embedding(client, title_str, model=model)
            new_done += 1

        if new_done % 1000 == 0:
            with open(path_embed, "wb") as f:
                pickle.dump(saved_data, f)
            print("Dumping at:", i, f"paper PMC={p}")

    with open(path_embed, "wb") as f:
        pickle.dump(saved_data, f)


def save_title_embedding(dir_dict, dir_temp):
    with open(os.path.join(dir_temp, "title_embedding.pkl"), "rb") as f:
        saved_data = pickle.load(f)
    saved_data = {paper: np.array(v) for paper, v in saved_data.items()}
    with open(os.path.join(dir_dict, "paper2embed.pkl"), "wb") as f:
        pickle.dump(saved_data, f)


"""
    ########################################
        Web of Science
    ########################################
"""


def WOS_get_author_last_name_and_paper(name, title, api_key, second_try=False):
    if not second_try:
        url = f'''https://api.clarivate.com/apis/wos-researcher/researchers?q=FirstName~"{name[0]}" AND LastName~"{name[1]}" AND TS~"{title}"'''
    else:
        url = f'''https://api.clarivate.com/apis/wos-researcher/researchers?q=LastName~"{name[1]}" AND TS~"{title}"'''
    response = requests.get(url, headers={"X-ApiKey": api_key})
    return response


def WOS_get_rid_author(rid, api_key):
    url = f"https://api.clarivate.com/apis/wos-researcher/researchers/{rid}"
    response = requests.get(url, headers={"X-ApiKey": api_key})
    return response


def WOS_get_author_data(name, paper_title, api_key, second_try=False):
    time.sleep(0.2)  # Assume 5 RPS rate limit.
    response = WOS_get_author_last_name_and_paper(name, paper_title, api_key, second_try)

    if response.status_code != 200:

        if response.status_code == 400:
            print("error code : " + str(response.status_code))
            return None
        print(response.status_code)
        print("error code : " + str(response.status_code))
        return response.status_code

    rid = None
    data = None
    try:
        if len(response.json()["hits"]) > 0:
            rid = response.json()["hits"][0]["rid"][0]
            time.sleep(0.2)  # Assume 5 RPS rate limit.
            data = WOS_get_rid_author(rid, api_key)
    except Exception as e:
        print("exception")
        print(e)
        traceback.print_exc()
        data = None
    return data


def _get_paper_min_date(paper2meta, pmcid):
    return min([d for d in paper2meta[pmcid]["pub-date"].values() if None not in d], default=(0, 0, 0))


def _get_best_paper_title(first_last_papers, ncheck=-1):
    try:
        if len(first_last_papers[1]) > 0:
            return sorted(first_last_papers[1])[ncheck][1]
        else:
            return sorted(first_last_papers[0])[ncheck][1]
    except:
        return "*"


def _download_from_WOS(author_titles, last_author2WOS, path_WOS, api_key, second_try, verbose=False):
    n_new = 0
    for i, name in enumerate(author_titles):
        if name in last_author2WOS:  # Acquired author meta from WOS, thus skipped.
            continue
        dat = None
        if not second_try:
            paper_title = _get_best_paper_title(author_titles[name])
        else:
            paper_title = _get_best_paper_title(author_titles[name], ncheck=-2)
        dat = WOS_get_author_data(name, paper_title, api_key, second_try)

        if isinstance(dat, numbers.Number):
            print(f"error code : {dat}")
            break

        if dat is not None:
            last_author2WOS[name] = dat.json()  # Only create author key if dat is not None.
            n_new += 1
            if "metricsAllTime" in dat.json() and verbose:
                print(f"n_new #{n_new} {name} {dat.json()['metricsAllTime']['hIndex']}")
        else:
            print(f"n_new #{n_new} not found: {name}")

        if n_new % 500 == 0:
            with open(path_WOS, "wb") as f:
                pickle.dump(last_author2WOS, f)

    with open(path_WOS, "wb") as f:
        pickle.dump(last_author2WOS, f)
    print(f"{len(last_author2WOS)/len(author_titles)*100:.2f}% of last authors WOS meta collected.")
    print(f"{n_new} authors added in this iteration.")


def get_author_meta_from_WOS(dir_dict, api_key, second_try=False, verbose=False):
    with open(os.path.join(dir_dict, "paper2last_author.pkl"), "rb") as f:
        paper2last_author = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2first_author.pkl"), "rb") as f:
        paper2first_author = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2meta.pkl"), "rb") as f:
        paper2meta = pickle.load(f)
    with open(os.path.join(dir_dict, "paper2full_title.pkl"), "rb") as f:
        paper2full_title = pickle.load(f)
    last_author2paper = reverse_dict_val(paper2last_author)
    first_author2paper = reverse_dict_val(paper2first_author)
    # last_author2paper = {k: v for k,v in last_author2paper.items() if k not in last_author2hIndex}

    author_titles = {}
    for au in last_author2paper:
        l_paper_titles = [(_get_paper_min_date(paper2meta, p), paper2full_title[p]) for p in last_author2paper[au]]
        if au in first_author2paper:
            f_paper_titles = [(_get_paper_min_date(paper2meta, p), paper2full_title[p]) for p in first_author2paper[au]]
        else:
            f_paper_titles = []
        author_titles[au] = [f_paper_titles, l_paper_titles]

    # Get (partially) saved data.
    path_WOS = os.path.join(dir_dict, "last_author2WOS.pkl")
    if os.path.exists(path_WOS):
        with open(path_WOS, "rb") as f:
            last_author2WOS = pickle.load(f)
    else:
        last_author2WOS = dict()
    _download_from_WOS(author_titles, last_author2WOS, path_WOS, api_key, second_try, verbose=verbose)


def save_last_author2hIndex(dir_dict):
    with open(os.path.join(dir_dict, "last_author2WOS.pkl"), "rb") as f:
        last_author2WOS = pickle.load(f)
    last_author2hIndex = {k: v["metricsAllTime"]["hIndex"] for k, v in last_author2WOS.items() if "metricsAllTime" in v}
    with open(os.path.join(dir_dict, "last_author2hIndex.pkl"), "wb") as f:
        pickle.dump(last_author2hIndex, f)


def transform_rating(ratings):
    avg = np.mean(ratings)
    if avg <= -2 / 5:
        return -1
    elif avg >= 2 / 5:
        return 1
    return 0


def check_dataset_err(dataset):
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


def save_finetune_file(init_dict, dir_batch, file_name):
    """
    Args:
        init_dict: openai object.
        file_name: File name of said training data.
        dir_batch: Training csv and saved json file (that will be sent to ChatGPT later).
    """
    setences = []
    df = pd.read_csv(os.path.join(dir_batch, f"{file_name}.csv"), sep=",")
    sentence_list = []
    ratings_list = []
    avg_rating_list = []

    for index, row in df.iterrows():
        sent = row["sentences"]
        ratings = np.array(row.drop("sentences"))
        sent = sent.replace("(|)", "").replace("[]", "")
        sentence_list.append(sent)
        avg_rating_list.append(transform_rating(ratings))
        ratings_list.append(ratings)

    ratings_list = np.array(ratings_list)
    avg_rating_list = np.array(avg_rating_list)

    with open(os.path.join(dir_batch, f"{file_name}.json"), "w") as f:
        for sent, rate in zip(sentence_list, avg_rating_list):
            dict_ = {}
            list_dicts = []
            list_dicts.append({"role": "system", "content": init_dict["system_prompt"]})
            list_dicts.append({"role": "user", "content": init_dict["user_prompt"] + sent})
            list_dicts.append({"role": "assistant", "content": str(rate)})
            dict_["messages"] = list_dicts

            line = str(json.dumps(dict_)) + " \n"
            f.write(line)

    # Load the dataset we just saved.
    with open(os.path.join(dir_batch, f"{file_name}.json"), "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    check_dataset_err(dataset)


def send_finetune_job(init_dict, model, dir_batch, file_name, hyperparams=None):
    job = init_dict["client"].files.create(file=open(os.path.join(dir_batch, f"{file_name}.json"), "rb"), purpose="fine-tune")
    print(f"Training file created:\n{job}")
    time.sleep(3)
    if hyperparams is None:
        hyperparams = {"learning_rate_multiplier": 0.5, "batch_size": 125, "n_epochs": 6}
    init_dict["client"].fine_tuning.jobs.create(training_file=job.id, model=model, hyperparameters=hyperparams)
    print(f"Finetuning job sent with hyperparameters:\n{hyperparams}")


def make_batches(init_dict, model, path_in, path_out, batch_size):
    batch_files = []
    list_todo = []
    list_sentences = []
    with open(os.path.join(path_in, "sentences2rate-CGPT.txt"), "r") as f:
        for index, line in enumerate(f):
            list_todo.append(index)
            list_sentences.append(line.strip())

    list_of_batch = []
    while True:
        if (len(list_of_batch) + 1) * batch_size < len(list_todo):
            list_of_batch.append(list_todo[len(list_of_batch) * batch_size : (len(list_of_batch) + 1) * batch_size])
        else:
            list_of_batch.append(list_todo[len(list_of_batch) * batch_size :])
            break

    for batch_todo in range(len(list_of_batch)):
        tasks = []
        for index_sent in list_of_batch[batch_todo]:

            citation_text = list_sentences[index_sent]
            messages = [{"role": "system", "content": init_dict["system_prompt"]}]
            messages.append({"role": "user", "content": init_dict["user_prompt"] + citation_text})
            task = {
                "custom_id": f"task-{batch_todo}-{index_sent}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0.01,
                    "messages": messages,
                },
            }
            tasks.append(task)

        file_name = os.path.join(path_out, f"sentiment_batch_{batch_todo}.jsonl")
        with open(file_name, "w") as file:
            for obj in tasks:
                file.write(json.dumps(obj) + "\n")

        batch_files.append(init_dict["client"].files.create(file=open(file_name, "rb"), purpose="batch"))

        # elements in list_of_batch[i] are indices for list_sentences, corresponding to "sentences2rate-CGPT.txt"
        batch_dict = {"batch_files": batch_files, "list_of_batch": list_of_batch, "list_sentences": list_sentences}

        with open(os.path.join(path_out, "batch_dict.pkl"), "wb") as f:
            pickle.dump(batch_dict, f)

        return batch_dict


def creat_batch_jobs(api_key, model, dir_sent, dir_batch, batch_size=49999):
    init_dict = CGPT_init(api_key)
    # Save jsonl files and batch_dict to /batch.
    batch_dict = make_batches(init_dict, model, dir_sent, dir_batch, batch_size)
    for x in batch_dict["batch_files"]:
        init_dict["client"].batches.create(input_file_id=x.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f'{len(batch_dict["batch_files"])} batches created.')


def process_batch_outputs(dir_in, dir_out):
    """
    batch output file names: batch_SOME_IDENTIFIER_STRING_output.jsonl
    "custom_id": f"task-{batch_todo}-{index_sent}"
        index_sent is row index corresponding to

    Save (& Return)
    ----
    - row2rate (dict):
        key: row num (same as sentrow2edgeinfo)
        val: rating
    """
    # Find batch files
    files = [file for file in os.listdir(dir_in) if file.startswith("batch_") and file.endswith("_output.jsonl")]
    row2rate = dict()
    for file in files:
        with open(os.path.join(dir_in, file), "r") as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            row_id = result["custom_id"].split("-")[-1]
            ans = result["response"]["body"]["choices"][0]["message"]["content"]
            row2rate[int(row_id)] = ans

    with open(os.path.join(dir_out, "row2rate.pkl"), "wb") as f:
        pickle.dump(row2rate, f)
    return row2rate


def prepare_author_names(dir_dict, dir_out):
    with open(os.path.join(dir_dict, "paper2last_author.pkl"), "rb") as f:
        paper2last_author = pickle.load(f)
    last_author2paper = reverse_dict_val(paper2last_author)
    df = pd.DataFrame(data={x: [None for _ in last_author2paper] for x in ["ID", "First Name", "Last Name"]})

    for i, name in enumerate(last_author2paper):
        df.at[i, "ID"] = i
        df.at[i, "First Name"] = name[0]
        df.at[i, "Last Name"] = name[1]

    df.to_csv(os.path.join(dir_out, "last_author_names.csv"), index=False)


def save_author_gender(dir_dict, dir_genderAPI):
    with open(os.path.join(dir_dict, "paper2last_author.pkl"), "rb") as f:
        paper2last_author = pickle.load(f)
    last_author2paper = reverse_dict_val(paper2last_author)
    df = pd.read_csv(os.path.join(dir_genderAPI, "gender-API.csv"))
    # Gender (str), accuracy (int), sample size (int).
    last_author2gender_info = {au: [None, None, None] for au in last_author2paper}
    # Gender field can be "unknown" or "", and in these two cases the value would be ["", -1, -1]
    for row in df.itertuples(index=False):
        au = (row.FirstName, row.LastName)
        if au not in last_author2gender_info:
            raise Exception(f"Gender API csv contains {au} that is not in <paper2last_author>.")
        if row.ga_gender == "unknown" or pd.isnull(row.ga_gender):
            tmp = ["", -1, -1]
        elif row.ga_gender in {"male", "female"}:
            ge = "man" if row.ga_gender == "male" else "woman"
            tmp = [ge, int(row.ga_accuracy), int(row.ga_samples)]
        else:
            raise Exception(f"Unknown gender {row.ga_gender} in gender API csv.")
        last_author2gender_info[au] = tmp

    with open(os.path.join(dir_dict, "last_author2gender_info.pkl"), "wb") as f:
        pickle.dump(last_author2gender_info, f)