from mqdq import babble, rhyme  # type:ignore
import pandas as pd  # type:ignore
import re
from mqdq.rhyme_classes import Line, LineSet  # type:ignore
from typing import Callable

# LOGGING SETUP
import logging

logger = logging.getLogger("embeddings_umap")
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] > %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

# MODULE CONSTANTS
XX = re.compile("..")


def _line_to_aug_syls(l: Line) -> list[str]:
    """
    Convert a line into a List of strings. Each token is one syllable. To this
    we add metadata for length, pauses, ictus, stress, and elision.

    Args:
        l: Line to convert

    Returns:
        The result.
    """
    final = []
    syl_feats = []
    for w in l:
        syls = re.findall(XX, str(w.mqdq["sy"]))
        for i, s in enumerate(syls):

            # Since we enumerate the MQDQ syllable string, we shouldn't get any
            # empty syllables (lost by elision) which can appear in my +syls+ List
            # in the Word object.

            syl_feats.append(w.syls[i].translate(rhyme.DEFANCY).lower())
            if s[1] == "A":
                # Ictus
                syl_feats.append("A")
            if s[1].isupper():
                # Long
                syl_feats.append("L")
            if i == w.stress_idx:
                # Accent / Stress
                syl_feats.append("S")

            if i + 1 == len(syls):
                # end of a word. Because these are OR'd it's possible for a
                # metron with two breves to contain two pauses (eg CM and CF).
                if w.mqdq.has_attr("wb"):
                    if w.mqdq["wb"] == "CM":
                        # strong caesura (end of foot after arsis)
                        syl_feats.append("SC")
                    elif w.mqdq["wb"] == "DI":
                        # diaeresis (end of foot == end of word)
                        syl_feats.append("DI")
                    elif w.mqdq["wb"] == "CF":
                        # weak caesura (end of foot after first breve in a dactyl)
                        syl_feats.append("WC")
                if w.mf == "SY":
                    # Elision aka synalepha
                    syl_feats.append("SY")

            final.append("+".join(syl_feats))

            del syl_feats[:]

    return final


def lines_to_aug_sylstream(ll: LineSet) -> list[str]:
    """
    Convert an mqdq.rhyme.LineSet into a (long) List of strings. Each token is
    one syllable with metadata for length, pauses, ictus, stress, and elision.
    See line_to_aug_syls(). This is a mapper suitable for use in chunk_lineset()
    (cf.)

    Args:
        ll: Lines to convert

    Returns:
        The result.
    """
    stream = []
    for l in ll:
        ss = _line_to_aug_syls(l)
        ss.append("EOL")
        stream.extend(ss)
    return stream


def _chunk_lineset(
    ll: LineSet,
    mapper: Callable[[LineSet], list[str]],
    sz: int,
    step: int,
    name: str = "",
    author: str = "",
    strict: bool = True,
    raw_bookrefs: bool = False,
) -> pd.DataFrame:

    """
    Take a LineSet and a Mapper function, and chunk the lines using a rolling
    window. Each line is converted to a list of tokens with the Mapper, which
    are then rejoined with spaces into a single string (pandas doesn't like
    lists in columns)

    Args:
        ll: lines to chunk
        mapper: function to run on each line
        sz: size (in lines) of each chunk
        step: step to advance rolling window
        name: entry in Work column
        author: entry in Author column
        strict: if True discard final chunk if less than sz
        raw_bookrefs: if True use line indices, otherwise try to parse a
            Classics style book:line ref as a string

    Returns:
        DataFrame with columns:
            - Chunk, the string
            - Raw, the BeautifulSoup list of Tags
            - Bookref, the start of the chunk
            - Work, work name
            - Author, Author
    """

    if step > sz:
        raise ValueError("Step cannot be greater than chunksize.")
    chunk_ary = []
    br_ary = []
    for idx in range(0, len(ll) - sz, step):
        chunk = ll[idx : idx + sz]
        if len(chunk) < sz and strict:
            break
        chunk_ary.append(chunk)
        if not raw_bookrefs:
            try:
                # the MQDQ line is the parent of the first word's MQDQ attribute
                l = ll[idx][0].mqdq.parent
                if l.parent.name == "division":  # type: ignore
                    book = str(l.parent["title"])  # type: ignore
                else:
                    book = ""
                ln = str(l["name"])
                br = book + ":" + ln
            except KeyError:
                br = "<??>"
            br_ary.append(br)
        else:
            br_ary.append(str(idx))

    df = pd.DataFrame()
    df["Chunk"] = [(" ").join(mapper(c)) for c in chunk_ary]
    df["Raw"] = chunk_ary
    df["Bookref"] = br_ary
    if name:
        df["Work"] = name
    if author:
        df["Author"] = author

    return df


def chunked_dataset(
    corpus: dict,
    mapper: Callable[[LineSet], list[str]],
    chunk: int = 64,
    step: int = 4,
    rand: int = 42,
    return_raw=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Wrapper method to convert the whole corpus to a test/train split DataFrame
    suitable for passing on to TF. Each work in the corpus will have 10% of the
    lines removed for validation, which are returned in a separate DataFrame
    (with the same columns). Chunks from the train set will always be
    contiguous, so no chunks with lines from before and after the holdout lines.

    Args:
        chunk: size of each chunk
        step: step size for rolling window (must be <= chunk)
        corpus: the corpus dict as defined here (not portable sorry)
        mapper: the Mapper lambda that will be used by the chunking functions
        rand (default=42): random seed for shuffling

    Returns:
        a Tuple of two DataFrames (train,test) with columns:
            - Chunk, the string
            - Raw, the BeautifulSoup list of Tags
            - Bookref, the start of the chunk
            - Work, work name
            - Author, Author
            - Factor, alphabetically sorted integer Factor for Author
    """
    train_raw = []
    holdout_raw = []
    for label in corpus.keys():
        fn, author, work, bab = corpus[label]
        if not bab:
            logger.debug(f"Creating Babbler {label}")
            bab = babble.Babbler.from_file(*fn)
            corpus[label] = (fn, author, work, bab)
        else:
            logger.debug(f"Re-using Babbler {label}")

        # pull out 10% of the lines for the final testing
        holdout_size = len(bab._syl_source()) // 10
        logger.debug(
            "Holding out %d lines from %d" % (holdout_size, len(bab._syl_source()))
        )
        # unscientifically, take it out from the middle
        midpoint = len(bab._syl_source()) // 2

        # chunk pre- and post-holdout separately so we don't have any chunks
        # with mixed content
        t_ll_pre = bab._syl_source()[:midpoint]
        t_ll_post = bab._syl_source()[midpoint + holdout_size :]
        t_df_pre = _chunk_lineset(
            t_ll_pre, mapper, chunk, step, name=work, author=author
        )
        t_df_post = _chunk_lineset(
            t_ll_post, mapper, chunk, step, name=work, author=author
        )
        t_df = pd.concat([t_df_pre, t_df_post])

        h_ll = bab._syl_source()[midpoint : midpoint + holdout_size]
        h_df = _chunk_lineset(h_ll, mapper, chunk, step, name=work, author=author)
        logger.debug(f"Got {len(t_df)} training chunks and {len(h_df)} holdout chunks.")
        train_raw.append(t_df)
        holdout_raw.append(h_df)

    train_df = pd.concat(train_raw)
    holdout_df = pd.concat(holdout_raw)
    train_df = train_df.sample(frac=1, random_state=rand).reset_index(drop=True)
    train_df["Factor"], _ = train_df.Author.factorize(sort=True)
    holdout_df = holdout_df.sample(frac=1, random_state=rand).reset_index(drop=True)
    holdout_df["Factor"], _ = holdout_df.Author.factorize(sort=True)
    if not return_raw:
        train_df = train_df.drop(["Raw"], axis=1)
        holdout_df = holdout_df.drop(["Raw"], axis=1)
    return (train_df, holdout_df)
