import argparse
import pandas as pd
import math
import inspect
import random
import spacy
from spacy.matcher import Matcher
import time

RATIO = 0.1
BLACKLIST= ["determiner", "punctuation", "particle"]

# stats builder 2
def get_stats(orig_doc:spacy.tokens.doc.Doc):
    """
    Returns the number of unique POS tags in the doc as (POS, Num, longPOS) Dataframe.
    """
    stats = dict()
    for token in orig_doc:
        try:
            stats[token.pos_] = stats[token.pos_] + 1
        except KeyError:
            stats[token.pos_] = 0
    stats_df = pd.DataFrame(stats.items(), columns=["POS", "Num"])
    stats_df["longPOS"] = stats_df["POS"].apply(lambda x:spacy.explain(x))
    return stats_df

# pattern builder 2
def get_patterns(stats_df:pd.DataFrame, blacklist:list[str]=[]):
    """
    Returns a list of lists of dictionary of POS and number of desired matches
    from a given Dataframe of (POS, Num, longPOS), after removing the blacklisted POS tags.
    """
    whitelist = stats_df[~stats_df["longPOS"].isin(blacklist)]
    return [[{
            "POS": pos,
            "OP" : f"{{,{num}}}" # each pattern is its own list
        }]
        for pos, num in zip(whitelist["POS"], whitelist["Num"] )]

def get_matches(patterns, matcher, doc, ratio=1.0):
    """
    Use a for loop to get the matches for each pattern.
    Take RATIO of each matches and replace them.
    """
    matches = []
    for pattern in patterns:
        matcher.add("PATTERN", [pattern])
        match = matcher(doc)
        matches.extend(
            random.sample(match, math.ceil(ratio * len(match)))
        )
        matcher.remove("PATTERN")

    # sort matches 

    return sorted(matches, key=lambda x:x[1])

# pasted again
def replace_word(orig_doc, matches):
    """
    Returns a string where the matches have been replaced by the corresponding POS tag information.
    """
    text = ''
    replacements =[]
    buffer_start = 0
    for _, match_start, _ in matches:
        if match_start > buffer_start:  # If we've skipped over some tokens, let's add those in (with trailing whitespace if available)
            text += orig_doc[buffer_start: match_start].text + orig_doc[match_start - 1].whitespace_
        
        token = orig_doc[match_start]
        tag = spacy.explain(token.tag_)
        pos = tag.split(',')[0]
        replacement = "{{{}}}"

        text += replacement + token.whitespace_ # Replace token, with trailing whitespace if available
        buffer_start = match_start + 1
        replacements.append((pos, tag))
    text += orig_doc[buffer_start:].text
    return text, replacements

def check_POS(inp, sug, nlp):
    doc = nlp(inp)
    return doc[0].pos_ == sug

def madlib():

    # start= time.monotonic()

    df = pd.read_csv("archive/cnn_dailymail/test.csv")
    nlp = spacy.load("en_core_web_sm")

    # choose a random article
    idx = random.randint(0, len(df)-1)
    text = df.iloc[idx]['article']
    text = df.iloc[idx]['highlights']
    doc = nlp(text)
    stats_df = get_stats(doc)
    matcher = Matcher(nlp.vocab)
    patterns = get_patterns(stats_df= stats_df, blacklist= BLACKLIST)
    matches = get_matches(patterns, matcher, doc, ratio=RATIO)

    blanked_txt, replacements = replace_word(doc, matches)
    blanked_txt = blanked_txt.replace(". ", ".\n").replace("? ", "?\n")
    brief = [x[0] for x in replacements]
    verbose = [x[1] for x in replacements]
   
    with open("new.txt", "w") as f:
        f.write(str(replacements))
        f.write("\n*******************\n")
        f.write(blanked_txt)
        f.write("\n*******************\n")
        f.write(blanked_txt.format(*brief))
        f.write("\n*******************\n")
        f.write(blanked_txt.format(*verbose))
        
        # f.write(str(time.monotonic()-start))
    return brief, verbose, blanked_txt.format(*brief), blanked_txt.format(*verbose),
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose. Switch between brief and verbose hints for each adlib.", action="store_true")
    args = parser.parse_args()
    a, b, c, d = madlib()
    if args.verbose:
        print(d)
    else:
        print(c)

    for e in a:
        inp = input(f"Give a {a}.")
        while check_POS(inp, e, nlp):
            print("Try again.")
            inp = input(f"Give a {a}.")
            

if __name__ == "__main__":
    main()