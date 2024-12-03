import os
import re
import pickle
import tqdm
import nltk
from lxml import etree

from helper_functions import flatten_list


NS = {"xml": "http://www.w3.org/XML/1998/namespace"}  # namespace for @xml:lang

"""
    ########################################
        xml parsing notes:
        There are 8 files that use <collab> to store author data;
        I modify them manually to fit the current parsing scheme
        said 8 PMCID (xml files): 3433679, 3923972, 5441062, 8315276, 9515132, 3418423, 5752703, 6519344

        use 8419223.xml as template (2-author) to modify the collab types below:
        <contrib-group>
            <contrib contrib-type="author">
                <name>
                    <surname>PLACEHOLDER</surname>
                    <given-names>PLACEHOLDER</given-names>
                </name>
                <xref ref-type="aff" rid="aff1">
                    <sup>1</sup>
                </xref>
            </contrib>
            <contrib contrib-type="author">
                <name>
                    <surname>PLACEHOLDER</surname>
                    <given-names>PLACEHOLDER</given-names>
                </name>
                <xref ref-type="aff" rid="aff2">
                    <sup>2</sup>
                </xref>
            </contrib>
        </contrib-group>
        <aff id="aff1">
            <sup>1</sup>
            <institution>PLACEHOLDER</institution>,
            <addr-line>PLACEHOLDER</addr-line>, 
            <country>PLACEHOLDER</country>
        </aff>
        <aff id="aff2">
            <sup>2</sup>
            <institution>PLACEHOLDER</institution>, 
            <addr-line>PLACEHOLDER</addr-line>, 
            <country>PLACEHOLDER</country>
        </aff>
    ########################################
"""


"""
    ########################################
        Make Functions
        make constants and initialize stuff
    ########################################
"""


def make_tags_exclude():
    return {
        "fig",  # figure, including caption
        "table",  # table
        "table-wrap",  # also table
        "table-wrap-foot",  # stuff below the table (i.e., caption/footnote); also part of table
        "fn",  # footnote; could be footnote in table
        "table-fn",  # e.g., superscript in table; under xref;
        # "table-fn"'s corresponding footnote in e.g., table-wrap-foot
    }


def make_unicode_dashes():
    # all 26 unicode dashes https://www.fileformat.info/info/unicode/category/Pd/list.htm
    dashes = r"\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015"
    dashes += r"\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u2E5D"
    dashes += r"\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D\u10EAD"
    return dashes


def make_unicode_minuses():
    # all 30 unique unicode characters containing "minus"
    # https://www.fileformat.info/info/unicode/char/search.htm?q=minus&preview=entity
    minuses = r"\u002D\u00B1\u02D7\u0320\u0F2A\u2052\u207B\u208B"
    minuses += r"\u2212\u2213\u2216\u2238\u2242\u2296\u229F\u2756\u2796\u293C"
    minuses += r"\u2A22\u2A29\u2A2A\u2A2B\u2A2C\u2A3A\u2A41\u2A6C\uFE63\uFF0D\uE002D"
    return minuses


def make_DASHes():
    return make_unicode_dashes() + make_unicode_minuses()


DASH = make_DASHes()

"""
    ########################################
        Helper Functions
    ########################################
"""


def _replace_Element_with_text(elem, str_):
    # remove the Element, leaving only str_
    str_ += elem.tail or ""  # str_ + tail text
    parent = elem.getparent()
    if parent is not None:  # removal (of parent's subElement) happens at parent level
        sibling_prev = elem.getprevious()
        # in both below cases, we suffix str_ to elem's "head" (text before its tag)
        if sibling_prev is not None:  # if there's prev sibling, suffix str_ to its tail
            sibling_prev.tail = (sibling_prev.tail or "") + str_
        else:  # if there's no prev sibling, suffix str_ to parent's text
            parent.text = (parent.text or "") + str_
        parent.remove(elem)  # remove the Element


def _is_path_no_tag_p(el_path, tags):
    """
    parse el's path as returned by tree.getelementpath(p)
    return True if there's no t in-between the last p and "body", for all t in tags
    """
    temp = el_path.split("/")
    if temp[0] != "body" or re.match(r"p(?:\[[0-9]+\])?$", temp[-1]) is None:
        raise Exception(f'assumption not met: {el_path} does not follow "body/.../p[]" format')
    else:
        for x in temp[1:-1]:  # in-between
            for t in tags:
                if re.match(rf"{t}(?:\[[0-9]+\])?$", x):  # True if found t in-between
                    return False
        return True


def _is_path_parent_sup_xref(el_path):
    """
    parse el's path as returned by tree.getelementpath(el)
    return True if the immediate parent/ancestor of xref has a tag of <sup>

    toy example of el that return False:
        "body/a/b/xref"
        "body/a/sup/b/xref"
    """
    temp = el_path.split("/")
    if temp[0] != "body" or re.match(r"xref(?:\[[0-9]+\])?$", temp[-1]) is None:
        raise Exception(f'assumption not met: {el_path} does not follow "body/.../xref[]" format')
    else:
        if re.match(r"sup(?:\[[0-9]+\])?$", temp[-2]):  # True if sup immediately before xref
            return True
        return False


def _p_processor1(tree, p, tags_exclude=None, encoding=None):
    """this generate passages ready to be processed with NLP stuff, like sentence splitting
    NOTE it modifies p in place
    Args
    ----
    - p (str): it contains xml tags such as <xref></xref>
    - tags_exclude (list of xpath str):
        we will remove p's Element whose tag is in this list
        make sure it's in f".//{tag_name}" format
        we use xpath because .findall() doesn't support "|" (as in it won't find anything)

    p.text yields the text before its 1st subElement (so if no subElement, it's complete text)
    if p has 1+ subElement (el; highest level),
    then el.tail is text between it and its next sibling (or the remaining text if no sibling)
    so if we concatenate p.text with el.tail, we get all the text (without the nested ones that is)
    make sure to consider None (no text)

    Return
    ------
    None if no citation found;
    pure text if citation found

    steps:
    1. find all citation instances and replace each with a specially encoded str
        f'◴{el.get("rid")}◷', where the stuff sandwiched is reference id,
        which corresponds to the reference id ref_id = el.get("id") in its parent function
        HOWEVER, some put multiple rids into the "rid" field, separated by " " (e.g., 4461310)
        the way we process them:
        [^{DASH}\w]+: [{DASH}\w]+ is how we select the rid (singular)
        so excluding the set means we select whatever that doesn't show up in the rid (singular)
        and the "whatever" includes " " delimiter
        since sometimes an rid (singular) includes dashes
        e.g., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4768648/#adma201502437-bib-0001
        we use [{DASH}\w]+ instead of [\w]+ to find the rid (singular)

        for one particular type, we will use f'◴SUP{el.get("rid")}SUP◷' instead,
        replacing the whole "sup" Element:
            this is the citation as a superscript type (incl. those placed after punctuations)
            we will deal with them differently in _p_processor2()
            e.g., PMC8532114
            <sup><xref ref-type="bibr" rid="ref1">1</xref>−<xref ref-type="bibr" rid="ref5">5</xref></sup>
            tail: " copper is also prone to deactivation during CO2RR."
        for these after-punctuation superscript type,
        we will move them to the last sentence in _p_processor2()
    will use it later in _p_processor2()
    ASSUME there's no nested citation, as in bibr-"xref" nested inside another bibr-"xref"
    there shouldn't be such citations (like, why would anyone double cite??)
    if there's nested citation, they will be ignored (undercount),
    because we replace the outermost citation with said specially encoded str,
    losing all stuff inside this outermost citation

    2. replace certain types of Element based on prior literature practice (e.g., figure captions)
    with a space " "
    this means we will not save any of the text inside these tags
    we use " " because otherwise it may connect two words that otherwise shouldn't be connected;
    and if the two words are already separated, we have double spaces, which will be dealt with

    3. replace bulletpoints (not content, just the labels) with space " "
        define bulletpoints as "label" Element whose immediate ancestor is "list-item"
        we do this simply because there could be "i." or "1." period type bulletpoints
        and it'd be hard to split sentences correctly when there are these bulletpoints

    4. for others, we keep all plain text recursively (i.e., strip out all tags)
    (we separate each text field by a space,
    and then append a space to the end of the text to return)

    different subElement tags seen in 11461 xml files:
    there are 102 types of non-"xref" if we look at all descendants recursively
    """
    # use tostring() to serialize in order to get plain text (xml markups included)
    # p = etree.tostring(p, encoding="unicode")  # unencoded unicode string
    if encoding is None:
        encoding = "UTF-8"
    if tags_exclude is None:
        tags_exclude = {".//" + x for x in make_tags_exclude()}  # "//" means recursive

    # step 1: replace all citations with specially encoded str
    el = flag_nocite = object()  # flag for no citation found
    for el in p.findall(".//xref[@ref-type='bibr']"):  # recursive; all bibr xref tags (citations)
        if _is_path_parent_sup_xref(tree.getelementpath(el)):
            temp_rid = re.sub(rf"[^{DASH}\w]+", "SUP◷|◴SUP", el.get("rid"))
            _replace_Element_with_text(el, f"◴SUP{temp_rid}SUP◷")
        else:
            temp_rid = re.sub(rf"[^{DASH}\w]+", "◷|◴", el.get("rid"))
            _replace_Element_with_text(el, f"◴{temp_rid}◷")
    if el is flag_nocite:  # no citation found
        return None
    # step 2: replace all specified tag-type Element with a space " "
    for el in p.xpath("|".join(tags_exclude)):  # all specified tags
        _replace_Element_with_text(el, " ")
    # step 3: replace bulletpoints with a space " "
    for el in p.findall(".//list-item/label"):  # recursive
        _replace_Element_with_text(el, " ")
    # step 4: get all text content from p, stripping out all tags and such
    # recursive; Element text only; 1st encode into proper b-str and then decode
    temp_s_pp1 = " ".join(p.itertext(etree.Element)).encode(encoding).decode(encoding) + " "  # add a space after each p
    if re.search(rf"◴[{DASH}\w]+◷", temp_s_pp1, flags=re.A):  # see if it still has 1+ citation
        return temp_s_pp1
    else:  # no citation found after step 2-4
        return None


def _p_processor2(p_mod, pmcid=None):
    """
    processes output of _p_processor1(), which is modified <p> (a passage of text), thus the name

    remove whitespaces:
    https://stackoverflow.com/questions/1546226/

    properly format citation chains (i.e., the same group of citations), defined as:
        consecutive f"◴{whatnot}◷" pattern without any word characters inside
        we will reformat the chains:
            e.g., from f"◴{ref1}◷,◴{ref4}◷-◴{ref6}◷"
            to f"◴{ref1}|{ref4}|{ref5}|{ref6}◷"

    on citations:
    for ranged citations (i.e., num_a-num_b, or in our case marked by "◷−◴"):
        ASSUME the intended included citations are inside the internal citation indices
        e.g., ◴SUPref19SUP◷−◴SUPref21SUP◷ (PMC8532114) means ref19, ref20, ref21
    for post-sentence citations (after sentence-splitting; considering each sentence):
        if the sentence starts with a citaton marker:
            if the citation chain is followed by nothing:
                ASSUME this citation chain belongs in the previous sentence
            elif the citation chain is followed by 1 non-whitespace character:
                if the char is "." or ",":
                    # only found 1 (out of 11461) that has ",": PMC9413444
                    "lipofuscin, a waste deposit in neurons.105−107,"
                    which looks like a typo
                    ASSUME this citation chain belongs in the previous sentence
                else:
                    ASSUME this citation chain belongs in the current sentence
            elif the citation chain is followed by a space and an uppercase letter (e.g., "◷ A")
                or by a quotation mark w/ or w/o the space (e.g., "◷ "", "◷ '", "◷"", "◷'"):
                ASSUME this citation chain belongs in the previous sentence
            else:
                the citation chain is followed by a space and a lowercase letter (e.g., "◷ a")
                as well as other miscellaneous cases
                ASSUME this citation chain belongs in the current sentence

        else:
            ASSUME all citations belong in the current sentence
    """
    # step 1: process some special cases & cleaning
    # fix some cases where default sent_tokenize() splits when it shouldn't
    # default (pre-trained) sent_tokenize() can't identify Latin abbreviations common in papers
    # Looked up common abbreviations, but couldn't find what I have encountered,
    # such as et al. and cf.ref. 41 (PMC5932982).
    sent_list = re.sub(r"[\s]et[\s]+al\.|\setc\.", "", p_mod)  # remove " et al." & "etc."
    # replace citation indicators with " "; \u25F4 is ◴
    cite_ind = r"\b(cf\.[,\s]*|e\.g\.[,\s]*)?[Rr]ef[s]?\.[\s]?"  # ref., cf. ref., e.g. ref.
    cite_ind += r"|\b(cf\.[,\s]*|e\.g\.[,\s]*)(?![Rr]ef[s]?\.[\s]?)"  # cf., e.g. (w/o ref.)
    cite_ind += r"|\b[Ff]ig[s]?\.[\s]?"  # Fig.
    cite_ind += r"|\b[Ee]q[s]?\.[\s]?"  # Eq.
    cite_ind += r"|\bi\.e\.[\s]?"  # i.e.
    cite_ind += r"|\b[Ss]ec[t]?[s]?\.[\s]?"  # sec(t)
    sent_list = re.sub(cite_ind, " ", sent_list)
    # remove all redundant whitespaces (space, tab, newline, return, formfeed)
    sent_list = " ".join(sent_list.split())

    # step 2: clean up citations to have a homogeneous structure
    sent_list = _chainify(sent_list).strip()  # remove leading/trailing whitespaces

    # step 3: tokenize by sentence (i.e., sentence-splitting)
    sent_list = nltk.tokenize.sent_tokenize(sent_list)  # sent_list is now a list of sentences

    # step 4: move post-sentence citations to the previous sentences
    # if citation is at the beginning when i=0 (<p>'s first sentence),
    # ASSUME it's never a post-sent citation; so we skip i=0
    # i.e., PMC8178227 (in 11641 files)
    for i in range(1, len(sent_list)):
        temp = _get_post_sent(sent_list[i], pmcid, p_mod)
        if temp:
            sent_list[i - 1] += temp[0]  # add post-sentence citation back to the correct sentence
            sent_list[i] = temp[1]  # remove the post-sentence citation in the current sentence

    return sent_list


def _sent_d_processor(sent_d, cite_marker="✪"):
    """collect all sentences that have at least one citation
    the analysis is at the sentence level,
    but we also collect chain-level statistics,
    specifically, the 3rd item to output is a list of chain lengths.

    we keep track of where the chains are in a sentence with cite_marker

    Arg
    ---
    - sent_d (dict): val is list of sentences

    Return
    ------
    - sents (dict of sentence):
    - cites (dict of list of citation chains):
        each item in the list is a set of reference id (for a citation chain)
        len of list = num of chains; each item <-> each chain; each ref id <-> ref in chain
    sents and cites have the same length because we include only citation sentences
        the values are indices shared between the two dict
    """
    sents = dict()
    cites = dict()
    idx = 0
    for v in sent_d.values():  # v is a list of sentences for each <p>
        if v is not None:  # there's at least one citation chain inside the whole <p>
            for s in v:  # s is a sentence in the <p>
                m1 = re.findall(rf"◴(?:[{DASH}\w]+(?:◷\|◴)?)+◷", s, flags=re.A)  # all chains
                if m1:  # there's at least 1+ chain in sentence s
                    # get the modified sentence by replacing each chain with a citation marker
                    # if the chain is preceeded by a whitespace, then we replace normally
                    sents[idx] = re.sub(rf"(?<=[\s])◴(?:[{DASH}\w]+(?:◷\|◴)?)+◷", cite_marker, s, flags=re.A)
                    # if the chain is not preceeded by a whitespace, then we add one
                    sents[idx] = re.sub(rf"(?<![\s])◴(?:[{DASH}\w]+(?:◷\|◴)?)+◷", " " + cite_marker, sents[idx], flags=re.A)
                    # remove redundant spaces
                    sents[idx] = " ".join(sents[idx].split())
                    # get citation ids
                    cites[idx] = [set() for _ in range(len(m1))]
                    for i, c in enumerate(m1):  # c is a chain
                        m2 = re.findall(rf"◴SUP([{DASH}\w]+)SUP◷|◴([{DASH}\w]+)◷", c, flags=re.A)  # sole citations
                        cites[idx][i] = {x[0] if x[0] else x[1] for x in m2}
                    idx += 1
    return sents, cites


def _chainify(s):
    """
    find citations and return <s> but with
    structured group citations (i.e., each is a chain containing potentially multiple citations)
    return <s> untouched if no citations found

    also move citation chains that are preceded immediately by punctuations
    (this is for post-sentence citations;
    some non-sentence cases are dealt with earlier, e.g., "ref.◴")

    structured citation format (it expands dash-ed/minused citations):
    from: ◴abc123◷,◴abc125◷−◴abc127◷, ◴abc130◷
    to: ◴abc123◷|◴abc125◷|◴abc126◷|◴abc127◷|◴abc130◷

    Arg
    ---
    - s (str): a sentence returned by nltk.tokenize.sent_tokenize()
    """
    temp = re.finditer(rf"◴(?:[{DASH}\w]+(?:◷[&,;\s{DASH}]*◴)?)+◷", s, flags=re.A)  # find all citation chains
    _empty = object()
    first = next(temp, _empty)
    if first is _empty:
        return s  # no citations found; return as is
    else:
        s_return = ""  # a modified copy of <s>
        i_current = 0  # current "cursor" in s_return
        _al = r"a-zA-Z_"
        cite_list = [first.span()] + [m.span() for m in temp]
        for ci in cite_list:  # each ci is a citation chain
            temp = re.sub(
                rf"(◴[{_al}]*)([0-9]+)([{_al}]*◷)\s*[{DASH}]+\s*(◴[{_al}]*)([0-9]+)([{_al}]*◷)",
                _repl_expand,
                s[ci[0] : ci[1]],
            )  # expand (e.g., 3-5 -> 3,4,5)
            temp = re.sub(r"(◷)[&,;\s]*(◴)", _repl_replace, temp)  # structure (e.g., 1,2 -> 1|2)
            str_pre = s[i_current : ci[0]]  # str from i_current to chain start
            temp2 = re.search(r"[^\w\s]+$", str_pre)  # find punctuation just pre-cite
            if temp2:  # punctuation followed by chain <temp>
                # i_current cj[0] cj[1] ci[0] ci[1] (before switching punctuation and chain)
                # note cj[0] cj[1] in the comments are s indices, but are str_pre indices in code
                cj0 = temp2.start()
                s_return += str_pre[:cj0] + temp + str_pre[cj0:]  # last item is punctuation
            else:
                s_return += str_pre + temp
            i_current = ci[1]
        s_return += s[i_current:]
        return s_return


def _repl_expand(matchobj):
    # ASSUME the two citations have the same prefix and suffix
    # if not, expanded citations will end up having the same pre-/suf- fixes as the 1st citation
    pref, n1, suff, n2 = matchobj.group(1, 2, 3, 5)
    return "|".join([f"{pref}{n}{suff}" for n in range(int(n1), int(n2) + 1)])


def _repl_replace(matchobj):
    suff, pref = matchobj.group(1, 2)
    return f"{suff}|{pref}"


def _get_post_sent(s, pmcid, p_mod):
    """
    Arg
    ---
    - s (str): a sentence

    Return
    ------
    a len-2 tuple of post-sentence citation AND <s> but with said citation removed, if it is found;
    None if post-sent-citation not found

    beginning of <s> may look like:
    ◴SUPref3◷|◴SUPref5◷|◴SUPref6◷ Beyond, blah blah blah

    sometimes, the post-sent-citation may not be enclosed in <sup> (e.g., PMC4503976),
    even though the article (pdf) does have them in superscripts
    (e.g., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4503976/)
    I used
        if m.group().find("◴SUP") == -1 or m.group().find("SUP◷") == -1:
            print(f"NOTE found a post-sent-citation not enclosed in <sup> tag: {pmcid}")
    to find these articles, and decided to not detect SUP anymore as a result
    (since such exceptions exist, there's no point considering SUP at all)
    """
    # find 1st citation chain that starts at the very beginning of the sentence <s>
    m = re.match(rf"◴(?:[{DASH}\w]+(?:◷\|◴)?)+◷", s, flags=re.A)
    if m:
        if len(s) == m.end():  # the whole str <s> is a chain
            return (m.group(), "")  # 'tis a post-sent-citation
        elif len(s) == m.end() + 1:  # <s> = chain + 1 non-whitespace char
            if s[-1] in {".", ","}:
                return (m.group(), s[-1])  # 'tis a post-sent-citation
            else:
                print(f"NOTE {pmcid} has a 'chain + 1 non-space char' sentence | {s} | in | {p_mod}\n")
                return None  # 'tis NOT a post-sent-citation
        elif s[m.end()] == " " and s[m.end() + 1].isupper():
            return (m.group(), s[m.end() + 1 :])  # 'tis a post-sent-citation
        else:
            return None  # 'tis NOT a post-sent-citation
    return None  # 'tis NOT a post-sent-citation


def _replace_Element_with_text(elem, str_):
    # remove the Element, leaving only str_
    str_ += elem.tail or ""  # str_ + tail text
    parent = elem.getparent()
    if parent is not None:  # removal (of parent's subElement) happens at parent level
        sibling_prev = elem.getprevious()
        # in both below cases, we suffix str_ to elem's "head" (text before its tag)
        if sibling_prev is not None:  # if there's prev sibling, suffix str_ to its tail
            sibling_prev.tail = (sibling_prev.tail or "") + str_
        else:  # if there's no prev sibling, suffix str_ to parent's text
            parent.text = (parent.text or "") + str_
        parent.remove(elem)  # remove the Element


def _xml_parser(xml_path, article_types, article_year, ref_meta_tags):
    """
    Args & Kwargs
    -------------
    - xml_path (str): the path to the xml file, which is the article's content

    PMC OA XML Element:
    tree structure (of Element) of interest:
    attr structure of each child node:
        "test_key1":"test_val1a"/"test_val1b"/etc. (FILTER)
        text1: blah
        "test_key2":"test_val2a"/"test_val2b"/etc.
        text2: blah
    attribute name (key) is unique inside an Element, but may not be unique among Element's siblings
    a node may have multiple attributes
    blah is the text field of the node
    (FILTER) means we may need this element for filtering

    - sent_d (dict): val is a list of sentence, and None if no citations

    all uid types:
    ---code---
    uid_types = set()
    # for xml
    for x in key_info_all.values():
        uid_types = uid_types.union(x["uid"].keys())
    # for each xml's reference
    for x in key_info_all.values():
        for y in x["ref-list"].values():
            uid_types = uid_types.union(y.keys())
    ---code---
    uid_types =
    {'doi', 'publisher-manuscript', 'pii', 'manuscript', 'pmc', 'publisher-id', 'pmid', 'other'}
    {'null', 'coden', 'doi', 'pmcid', 'arxiv', 'pmid', 'other'}  # reference

    pmcid should be the same as pmc

    /article/front
        /journal-meta
            /journal-id: usually more than one
        /article-meta
            /title-group:
                /article-title
            /article-id: usually more than one
                "pub-id-type"|"pmid" (attr as key|val pair):
                (text): corresponding id
            /pub-date: usually more than one
                "pub-type"|"epub" (attr as key|val pair)
                /year: should be present; may use this, and nothing more specific
                /month
                /day
    /article/body
        /sec or /p: usually more than one
        some articles use /p (e.g., PMC10375595), some use /sec (e.g., PMC4208460);
        they seem to have pretty different tree structures,
        however, the texts should all be contained in /p's text field,
            so we recursively find it
            only highest level, because p can be nested inside p
            in other words, we find all p that are not nested inside p
        on sections: There are "custom" section titles (e.g., PMC4208460),
        so going by section titles to find p is a no-go

    /article/back/ref-list
            /ref: "id" attr val is the same used in the "body"
                1st element tag (examples from 11461 files; all are research/review-article):
                    'citation': {'pmc': 'PMC1305130.xml'}
                    'mixed-citation': {'pmc': 'PMC10008059.xml'}
                    'element-citation': {'pmc': 'PMC10009610.xml'}
                    above 3 are all proper citations; I didn't find any sig. diff.
                    'label': {'pmc': 'PMC10020085.xml'}
                        this doesn't contain ref info; instead it's just alternative to "id" attr
                    'note': {'pmc': 'PMC10103050.xml', 'pmc': 'PMC10037336.xml'}
                        it's not a proper ref; it does have its own "id" attr, alt to "id" attr
                        it looks like notes
                    2nd one in 'note' is rapid-communication
                /citation or mixed-citation or element-citation:
                NOTE apparently there are cases where ref-list has more reference than used in text
                e.g., PMC6536847 ref_id = B18 (#14 in the paper ref list, not seen in the paper)
                    (authors: should have either text or "/person-group", but not both)
                    (text): authors in text form
                    /person-group:
                        person-group-type | "author"
                        /name or /string-name (see below for why we don't use the rest):
                        {'name': 'PMC3695061.xml',
                        'string-name': 'PMC8372019.xml',
                        'collab': 'PMC3695061.xml',  # has value (not quite like names) but no /name
                        'etal': 'PMC3695061.xml',  # no value but has /name anyway, prolly means more authors
                        'x': 'PMC8960998.xml'}  # has value (ellipsis) and has /name, prolly means more authors
                            /surname: last name
                            /given-names: first name initials (some have periods, some don't)
                    /year
                    /volume
                    /fpage
                    /lpage
                    /pub-id: same struct as /article-meta/article-id; some don't have it

    Return
    ------
    - key_info (nested dict):
        if it has "not-en" key, it's not written in English
            per https://jats.nlm.nih.gov/archiving/tag-library/1.2/attribute/xml-lang.html,
            default for top-level tag <article> is "en"; also it's case insensitive
        if it has "not-article-type" key, it doesn't belong to research/review article
        if it has "not-ref-list" key, it has no reference

        "article-type":
        {'letter', 'case-report', 'other', 'article-commentary', 'discussion',
         'rapid-communication', 'correction', 'brief-report', 'review-article',
         'editorial', 'news', 'abstract', 'retraction', 'research-article'}
        "xml:lang":

        "article-title":
            str of the title

        "authors":
            key: author order (starting from 0); val: dict
            val["name"] = ("first name", "last name")
            val["aff"] = (article_year, affiliations)
            affiliations is a tuple of strings, each one being an affiliation

        "uid" (dict):
            "pmid": PubMed ID
            "pmc": PubMed Central ID (same as the xml fname)
            "doi": doi
            etc.
        "pub-year" (dict):
            e.g., "ppub", "epub", "pmc-release": year
        "ref-list" (nested dict):
            ref_id (key) | meta data (val; nested dict)
                "publication-type": as the key name implies
                "volume/year/fpage": volume num, publication year, and first page num
                f"pub-id-{type}": type is the type of uid, and the val is the uid
                f"authors-{type}" (nested dict): type should be "author"
                    key is idx of author (in the order shown in xml; 0,1,...)
                    val is dict of name type (key; e.g., "surname", "given-names") and text (val)
                    "surname" is last name; "given-names" is first name initials
                    NOTE since some references (el_cite) don't have either pub-id or person-group,
                    we'd need to extract the names ourselves from el_cite's text field
                    but the pure text could follow different format, making extraction very hard
                    so we will not consider these references
                # "source": journal abbreviation; unused because the Element in xml is bugged:
                    abbreviation may be wrongly split into multiple source Elements
                    resulting in weird sources
                    for example, in PMC=10008059, ref B1:
                        source[0]: Proc Natl Acad Sci U
                        source[1]: S
                        source[2]: A
        "sents" and "cites": both are dict
            cites[i] contains a list of citation chains appeared in sents[i] (a sentence str)
            sents[i] has an in-text citation marker for each chain


    """
    tree = etree.parse(xml_path)  # class=lxml.etree._ElementTree
    encoding = tree.docinfo.encoding  # encoding attr of xml declaration param
    root = tree.getroot()  # class=lxml.etree._Element

    article = root.xpath("/article")  # list of article Element; absolute path; rest are relative
    # journal_meta = root.xpath("/article/front/journal-meta")[0]
    # article_meta = root.xpath("/article/front/article-meta")[0]

    if len(article) != 1:  # should be exactly one Element
        raise Exception(f'{xml_path} has {len(article)} "article" Element, expecting exactly 1')

    article = article[0]
    key_info = dict()
    # attributes in article tag
    key_info["article-type"] = article.get("article-type")
    temp_lang = article.xpath("./@xml:lang", namespaces=NS)
    key_info["language"] = temp_lang[0] if temp_lang else "en"  # cuz "en" is default
    # filter by article-type because the other types have very weird element tags
    if key_info["article-type"] not in article_types:
        key_info["not-article-type"] = None  # doesn't belong to any of article_types
        return key_info
    if key_info["language"].lower() != "en":
        key_info["not-en"] = key_info["language"]  # not written in EN
        return key_info
    # reference section (back)
    if not article.xpath("back/ref-list/ref"):  # no reference: empty list is False
        key_info["not-ref-list"] = None  # has no reference
        return key_info

    key_info["uid"] = dict()
    key_info["pub-year"] = dict()
    key_info["pub-date"] = dict()
    key_info["ref-list"] = dict()

    # pmcid here is only used for debugging prints
    pmcid = os.path.basename(xml_path)
    if pmcid.startswith("PMC"):
        pmcid = pmcid[3:]

    for el in article.xpath("back/ref-list/ref"):
        ref_id = el.get("id")
        key_info["ref-list"][ref_id] = dict()

        cite_types = {"citation", "mixed-citation", "element-citation"}
        for el_cite in el:  # there should be only 1 Element
            if el_cite.tag in cite_types:
                # get ref meta data: publication-type (attr),
                # pub-id,
                # volume, year, fpage,
                # author last name and first initials,
                key_info["ref-list"][ref_id]["publication-type"] = el_cite.get("publication-type")
                for el_meta in el_cite:
                    if el_meta.tag == "pub-id":  # 1+ Element if pub-id exists
                        temp_str = f'pub-id-{el_meta.get("pub-id-type")}'
                        key_info["ref-list"][ref_id][temp_str] = el_meta.text
                    elif el_meta.tag in ref_meta_tags:
                        key_info["ref-list"][ref_id][el_meta.tag] = el_meta.text
                    elif el_meta.tag == "person-group":
                        temp_str = f'authors-{el_meta.get("person-group-type")}'
                        key_info["ref-list"][ref_id][temp_str] = dict()
                        for i, el_ppl in enumerate(el_meta):  # 1+ Element
                            # ASSUME only 1 tag: either name or string-name (not both)
                            # if wrong, then this part of code is broken
                            if el_ppl.tag == "name" or el_ppl.tag == "string-name":
                                key_info["ref-list"][ref_id][temp_str][i] = {x.tag: x.text for x in el_ppl}

    # meta-data section (front)
    # uid
    for el in article.xpath("front/article-meta/article-id"):
        key_info["uid"][el.get("pub-id-type")] = el.text
    # pub-type is one attribute name; there's also date-type
    for el in article.xpath("front/article-meta/pub-date"):
        year = el.xpath("year")
        month = el.xpath("month")
        day = el.xpath("day")
        if len(year):  # only if there's year Element in this pub-date Element
            key_info["pub-year"][el.get("pub-type")] = year[0].text
        year = int(year[0].text) if len(year) else None
        month = int(month[0].text) if len(month) else None
        day = int(day[0].text) if len(day) else None
        attr_ = el.get("pub-type") if el.get("pub-type") is not None else el.get("date-type")
        key_info["pub-date"][attr_] = (year, month, day)
    # title
    title_ = article.xpath("front/article-meta/title-group/article-title")
    if len(title_) == 1:
        key_info["article-title"] = title_[0].text
    elif len(title_) > 1:
        key_info["article-title"] = title_[0].text
        print(f'warning427 {key_info["uid"]} has more than 1 title: {title_} (used 1st one)')
    else:  # no title-group/article-title
        print(f'warning427 {key_info["uid"]} has no title-group/article-title')
        key_info["article-title"] = None

    # author & affiliations
    flag_au = True
    key_info["authors"] = dict()
    contrib_grp = article.xpath("front/article-meta/contrib-group")  # list
    if len(contrib_grp) == 0:
        print(f'warning427 {key_info["uid"]} has 0 contrib Element, authors acquisition skipped')
        flag_au = False

    if flag_au:
        rid2aff = dict()  # aff can be all "front/article-meta/aff" but also in "front/article-meta/contrib-group"
        # key: rid; val: str of whatever institution (the whole string, not just institution)
        # below subElement should all be affiliation (ref-type="aff" in contrib_grp)
        for aff in article.xpath("front/article-meta/aff") + flatten_list([c.findall("aff") for c in contrib_grp]):
            if aff.find("institution") is not None:  # type1: it has institution and such as the subElement
                for el in aff.findall(".//sup"):  # recursive; remove "sup" subElement (superscript)
                    _replace_Element_with_text(el, "")
                etree.strip_tags(aff, "*")  # remove all tags (but leave the content intact)
                rid2aff[aff.get("id")] = aff.text
            else:  # type2: just plain text with "label" subElement we don't care, so we remove the labels
                for el in aff.findall(".//label"):  # recursive; remove "label" subElement
                    _replace_Element_with_text(el, "")
                etree.strip_tags(aff, "*")  # remove all tags (but leave the content intact)
                rid2aff[aff.get("id")] = aff.text
            # if there's a 3rd type... well, too bad it's subsumed above ;p

        au_o = 0  # author order; assume the document order = in author order

        for au in flatten_list([c.findall("contrib") for c in contrib_grp]):
            if au.get("contrib-type") == "author":
                namae = au.find("name")  # there should just be one subElement, ignore the others if more
                if (
                    au.find("collab") is not None
                ):  # collab type skipped (could still have author names, e.g., PMC5441062; only 8/110K papers)
                    # in PMC5441062, the aff id ref subElement is in the author <contrib> Element that contains the collab
                    # it's too much work for a mere 8/110K papers
                    # I hand modified them 8 papers:
                    # if they don't have aff, I put their collab name as institution
                    # if they don't have authors, I put their collab name as first name and last name
                    # print(f'warning427 {key_info["uid"]}, below contrib Element is not human author, but collab, skipped:')
                    # print(etree.tostring(au, pretty_print=True).decode())
                    continue
                if namae is None:
                    namae = au.find("name-alternatives/name")
                    if namae is None:  # not collab, nor name-alternatives, skipped
                        print(f'warning427 {key_info["uid"]}, below contrib Element is neither collab nor name-alternatives, skipped:')
                        print(etree.tostring(au, pretty_print=True).decode())
                        continue
                tmp_fname = namae.find("given-names")  # there are a few cases where this tag is present, but no text (sole <given-names/>)
                tmp_lname = namae.find("surname")
                if tmp_fname is None or tmp_lname is None or tmp_fname.text is None or tmp_lname.text is None:  # incomplete name skipped
                    continue
                author_hashable = (tmp_fname.text, tmp_lname.text)
                if au.find("aff") is None:
                    # affiliation indices (need them to loop up when we find the actual affiliation)
                    xrefs = au.findall("xref")
                    key_info["authors"][au_o] = {
                        "name": author_hashable,
                        "aff": (article_year, tuple(rid2aff[x.get("rid")] for x in xrefs if x.get("rid") in rid2aff)),
                    }
                else:  # aff subElement in contrib Element
                    aff_str_list = []
                    for aff in au.findall("aff"):
                        if aff.find("institution") is not None:  # type1: it has institution and such as the subElement
                            for el in aff.findall(".//sup"):  # recursive; remove "sup" subElement (superscript)
                                _replace_Element_with_text(el, "")
                            etree.strip_tags(aff, "*")  # remove all tags (but leave the content intact)
                            aff_str_list.append(aff.text)
                        else:  # type2: just plain text with label subElement we don't care, so we remove the labels
                            for el in aff.findall(".//label"):  # recursive; remove "label" subElement
                                _replace_Element_with_text(el, "")
                            etree.strip_tags(aff, "*")  # remove all tags (but leave the content intact)
                            aff_str_list.append(aff.text)
                        # if there's a 3rd type... well, too bad it's subsumed above ;p
                    key_info["authors"][au_o] = {
                        "name": author_hashable,
                        "aff": (article_year, tuple(x for x in aff_str_list)),
                    }
                au_o += 1
            # else:  # commented out cuz it doesn't matter what other contrib Element is like
            #     print(f'warning427 {key_info["uid"]}, below contrib Element is not author:')
            #     print(etree.tostring(au, pretty_print=True).decode())

    # main text section (body); this is where we first start modifying the tree
    ps = article.findall("body//p")  # find all p Element in body Element
    # get only p that are at the topmost level, i.e., no more p in the ancestor
    # but also no other tags (as specified in temp_tags_exclude) in the ancestor
    temp_tags_exclude = make_tags_exclude()
    ps_topmost = [p for p in ps if _is_path_no_tag_p(tree.getelementpath(p), {"p", *temp_tags_exclude})]
    sent_d = {i: None for i in range(len(ps_topmost))}
    for i, p in enumerate(ps_topmost):
        try:
            kwargs = dict(tags_exclude=temp_tags_exclude, encoding=encoding)
            sent_d[i] = _p_processor1(tree, p, **kwargs)
            if sent_d[i] is not None:  # found citation in the <p>
                sent_d[i] = _p_processor2(sent_d[i], pmcid=pmcid)
        except ValueError:
            print(f"DEBUG {pmcid} p: {etree.tostring(p, encoding=encoding)}")
            raise

    key_info["sents"], key_info["cites"] = _sent_d_processor(sent_d)
    return key_info


def parse_all_xml_files(path_in, path_out, journal_year_lookup, article_types=None):
    """This function uses _xml_parser() to do individual parsing.
    Extract both sentence and meta info from xml files.
    Filter out articles based on the extracted info from xml:
        not written in EN
        not right type
        no citations in reference

    Args
    ----
    - path_in: directory containing all xml files to process
        ASSUME xml file uses either one of the following naming conventions:
        a. f"PMC{pmcid}.xml" (should be default)
        b. f"{pmcid}.xml" (but 'tis okay too)
    - path_out: directory containing the key_info_all pkl file

    Kwargs
    ------
    - article_types (set of str):
        types of articles to mark for filtering; default to research and review

    Intermediary
    ------------
    - <pmcid>: "pmc" in "uid" entry of the output dict
    """
    if article_types is None:
        article_types = set(["research-article", "review-article"])
    ref_meta_tags = set(["volume", "year", "fpage"])  # source is journal

    key_info_all = dict()
    keys_filtered = {"not-article-type", "not-en", "not-ref-list"}

    fname_pkl = "key_info_all.pkl"
    if fname_pkl in os.listdir(path_out):  # if the pkl file exists, load
        with open(os.path.join(path_out, fname_pkl), "rb") as f:
            key_info_all = pickle.load(f)
    # parse ALL xml files and save key_info_all (dict) as a pkl file
    i = 0  # count how many xml we are parsing
    # only process xml files f"{pmcid}.xml"
    files = [file for file in os.listdir(path_in) if file.endswith(".xml")]
    n_tot = len(files)
    n_filtered = 0
    for file in tqdm.tqdm(files):
        # exclude extension and prefix (if present), leaving only the numeral (pmcid)
        pmcid = file[3:-4] if file.startswith("PMC") else file[:-4]
        pmcid = int(pmcid)
        if pmcid in key_info_all:
            continue
        article_year = journal_year_lookup[pmcid][1]
        key_info_all[pmcid] = _xml_parser(os.path.join(path_in, file), article_types, article_year, ref_meta_tags)
        for k_f in keys_filtered:  # filter
            if k_f in key_info_all[pmcid]:
                del key_info_all[pmcid]
                n_filtered += 1
                break
        i += 1
        if i % 2000 == 0:
            with open(os.path.join(path_out, fname_pkl), "wb") as f:
                pickle.dump(key_info_all, f)
    with open(os.path.join(path_out, fname_pkl), "wb") as f:
        pickle.dump(key_info_all, f)
        print(f"<key_info_all> saved; size={len(key_info_all)} from {n_tot} xml files ({n_filtered:d} got filtered)")

    return key_info_all


def _get_pmcid_from_ref(ref_temp, uid_dicts):
    # ref_temp is a dict (uid_type:uid)
    # return pmcid when ref_temp doesn't have "pub-id-pmcid"
    # among the ref's uid_types, at least one of them is in the uid_types of the paper
    # (see the 3-condition list comprehension in generate_ref_stats())
    # so that's why the Exception is impossible to trigger
    if ref_temp.get("pub-id-pmid") in uid_dicts["pmid"].keys():
        return uid_dicts["pmid"][ref_temp["pub-id-pmid"]]
    elif ref_temp.get("pub-id-doi") in uid_dicts["doi"].keys():
        return uid_dicts["doi"][ref_temp["pub-id-doi"]]
    else:
        raise Exception("impossible")


def make_ref_stats(path_out, key_info_all, journal_year_lookup, journal_list):
    """process sentences to find some stats

    save ref_stats in path_out if not saved

    Args & Kwargs
    -------------
    - journal_year_lookup (dict): key is pmcid; val is len-2 list [journal (str), year (int)]

    paper stats:
    uid_lists (dict):
        key: uid_type
        value: a list of <k> type uid
    uid_dicts (dict): reverse look-up; excludes None
        key: uid_type
        value: a dict of uid:pmc

    Return & Save
    -------------
    ref_stats (dict): each paper has a list of reference papers; we store this info here
        same 1st level structure as key_info_all
        except the value dict contains all reference stats of that paper
            "jyf" (list): [journal (str), year (int), field (str)]
            "num_": scalar statistic
            "ref-list": same as key_info_all[pmc]["ref-list"] (dict whose val: dict (uid_type:uid))
                note that uid_type has a prefix "pub-id-"
                the difference is that:
                for items whose keys are in papers, we make sure pmcid is present in val
                because every paper in papers has a pmcid
            "ref_within": list of keys of "ref-list" that are in papers
            "cites": same as key_info_all[pmc]["cites"]
    """
    fname_pkl = "ref_stats.pkl"
    if fname_pkl in os.listdir(path_out):
        with open(os.path.join(path_out, fname_pkl), "rb") as f:
            return pickle.load(f)

    uid_types = ["pmid", "pmc", "doi"]
    uid_lists = dict()
    uid_dicts = dict()
    for u in uid_types:
        uid_lists[u] = [x["uid"].get(u) for x in key_info_all.values()]
        uid_dicts[u] = {x["uid"].get(u): y for y, x in key_info_all.items() if x["uid"].get(u) is not None}
        # print(f"doesn't have {u}: {len([x for x in uid_lists[u] if x is None])}")

    ref_stats = dict()
    fields = ["Neuroscience"]  # only feature one field; can be more, but then need to change other things like journal_list
    journal_sets = {fields[i]: set(journal_list) for i in range(len(fields))}
    for pmc, x in tqdm.tqdm(key_info_all.items()):
        ref_stats[pmc] = dict()
        temp_l = [f for f in journal_sets if journal_year_lookup[pmc][0] in journal_sets[f]]
        ref_stats[pmc]["jyf"] = journal_year_lookup[pmc] + temp_l
        # sentence processing: sentence-level analysis happens here
        ref_stats[pmc]["cites"] = dict()
        for k_sent in x["sents"].keys():
            ref_stats[pmc]["cites"][k_sent] = x["cites"][k_sent]
        ref_stats[pmc]["ref-list"] = x["ref-list"]
        ref_stats[pmc]["num_ref"] = len(x["ref-list"])  # num of reference
        # ref-list but among the papers; requires some form of reference identification:
        ref_stats[pmc]["ref_within"] = [
            i
            for i, d in x["ref-list"].items()
            if (
                (d.get("pub-id-pmid") in uid_dicts["pmid"].keys())
                or (d.get("pub-id-doi") in uid_dicts["doi"].keys())
                or (d.get("pub-id-pmcid") in uid_dicts["pmc"].keys())
            )
        ]
        # add pmc to ref-list if it doesn't have pmc; all papers have pmc so this should always work
        for i in ref_stats[pmc]["ref_within"]:
            if "pub-id-pmcid" not in ref_stats[pmc]["ref-list"][i]:
                ref_stats[pmc]["ref-list"][i]["pub-id-pmcid"] = _get_pmcid_from_ref(ref_stats[pmc]["ref-list"][i], uid_dicts)

        ref_stats[pmc]["num_ref_within"] = len(ref_stats[pmc]["ref_within"])

    with open(os.path.join(path_out, fname_pkl), "wb") as f:
        pickle.dump(ref_stats, f)
    return ref_stats
