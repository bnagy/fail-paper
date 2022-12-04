import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import re
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from bs4.element import Tag  # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageEnhance  # type: ignore
from typing import Iterable, Sequence

from mqdq import rhyme, rhyme_classes  # type: ignore


def split_pad_many(chunk: pd.Series, hmm: Sequence) -> list[list[list[int]]]:
    """
    Take many 1D attention heatmaps, and split and pad them to 18 syllables,
    according to the lines from the original poem chunks (to make them 2D with
    padding at the ends of lines)
    """
    ll = []
    if len(chunk) != len(hmm):
        raise ValueError("Mismatched lengths!")
    for i, hm in enumerate(hmm):
        idx_ary = eol_indices(chunk[i])
        lines = [hm[idx_ary[i] : idx_ary[i + 1]] for i in range(len(idx_ary) - 2)]
        Z = np.zeros((len(lines), 18))
        for i, row in enumerate(lines):
            Z[i, : len(row)] += row
        ll.append(Z)  # type:ignore
    return ll  # type:ignore


def eol_indices(chunk: str) -> Sequence:
    sary = chunk.split(" ")
    idx_ary = np.where(np.array(sary) == "EOL")[0]
    # build an array of indices that can be used as successive [from:to] indices
    # for python slices, so the first pair should start at 0 and the final one
    # should end at len(v_enc). Since all the EOL indices we have need to be
    # incremented by one anyway we hack around a bit to get the correct final
    # list
    idx_ary = np.concatenate([[-1], idx_ary, [len(sary)]])  # type:ignore
    return idx_ary  # type:ignore


def draw_thumbnails(
    heatmaps: Sequence,
    titles: Sequence,
    cols=16,
    w: float = 0,
    h: float = 0,
    show_idx: bool = True,
    fn: str = "",
    dpi=144,
) -> None:

    """
    matplotlib hackery to draw all the heatmaps at once. I hate matplotlib.
    """
    # Render
    # dumb trick to round up with integer division
    ROWS = -(-len(heatmaps) // cols)
    # keep the aspect ratio correct
    if w == 0:
        w = 12
    if h == 0:
        h = 3 * ROWS

    _, ax = plt.subplots(
        nrows=ROWS, ncols=cols, figsize=(w, h), dpi=dpi, facecolor=(1, 1, 1)
    )
    for i in range(len(heatmaps)):
        if show_idx:
            ax[i // cols][i % cols].set_title(f"{i}: {titles[i]}", fontsize=5)
        else:
            ax[i // cols][i % cols].set_title(f"{titles[i]}", fontsize=5)

        # imshow automatically scales the values to [0,1]. To create identical
        # images with PIL (elsewhere) we need to do that ourselves before
        # colormapping it.
        ax[i // cols][i % cols].imshow(heatmaps[i], cmap="jet")
        ax[i // cols][i % cols].axis("off")

    # if X//16 != 0 we will have blank spots in the last row so turn off those axes
    for i in range(len(heatmaps), cols * ROWS):
        ax[i // cols][i % cols].axis("off")

    if fn:
        plt.savefig(fn)
    return None


def _metron_length(syl_str: str) -> float:
    ml = 0
    ml += 1 * syl_str.count("A")
    ml += 1 * syl_str.count("T")
    ml += 1 * syl_str.count("X")
    ml += 1 * syl_str.count("b")
    ml += 1 * syl_str.count("c")
    return ml


def _partition_line_by_metrons(l: rhyme_classes.Line) -> list[tuple[str, int]]:

    final: list = []
    remainder_s = ""
    remainder_wl = 0.0
    for w in l:
        w = w.mqdq
        wl = _metron_length(w["sy"])
        if w.has_attr("mf") and w["mf"] == "SY":
            remainder_s = remainder_s + w.text + "_"
            remainder_wl += wl
        elif w.has_attr("mf") and w["mf"] == "PE":
            final[-1] = (final[-1][0] + " " + w.text, final[-1][1])
        else:
            final.append((remainder_s + w.text, wl + remainder_wl))
            remainder_s = ""
            remainder_wl = 0.0
    return final


def _add_title(im: Image.Image, title: str) -> Image.Image:
    # rotate, write in title, rotate back
    # (to get a sideways title)
    fnt = ImageFont.truetype("DejaVuSans.ttf", 20)
    rot = im.rotate(90, expand=True)
    d = ImageDraw.Draw(rot)
    d.text((30, 30), title, font=fnt, fill="white", anchor="lm")
    return rot.rotate(270, expand=True)


def attention_single(
    hmaps: Iterable,
    chunk: Sequence,
    titles: Sequence,
    idx: int,
    cmap: str = "jet",
    alpha=128,
    sampling=Image.Resampling.LANCZOS,
    fn: str = "",
) -> Image.Image:

    W, H = (50, 50)

    fnt_big = ImageFont.truetype("DejaVuSans.ttf", 14)
    fnt_small = ImageFont.truetype("DejaVuSans.ttf", 10)

    heatmap = (
        matplotlib.cm.get_cmap(cmap)(hmaps[idx] / hmaps[idx].max()) * 255  # type:ignore
    ).astype("uint8")
    im = Image.fromarray(heatmap)
    im = im.resize((heatmap.shape[1] * W, heatmap.shape[0] * H), sampling)
    white = Image.new(
        size=(heatmap.shape[1] * W, heatmap.shape[0] * H), color="#FFFFFF", mode="RGBA"
    )
    im.putalpha(alpha)
    white.alpha_composite(im)
    d = ImageDraw.Draw(white)

    """
    this part is all a bit gross :/
    """
    for i, l in enumerate(chunk[idx]):
        ml = _partition_line_by_metrons(l)
        offset = (heatmap.shape[1] - 18) * W
        ypos = i * H + H / 2
        for s, width in ml:
            # offset so far, then middle of this 'cell'
            xpos = offset + (width * W / 2)
            d.text((xpos, ypos), s, font=fnt_big, fill="black", anchor="mm")
            offset += width * W

        ypos += H / 3
        xpos = (heatmap.shape[1] - 18) * W
        xpos += W / 2
        for w in l:
            # array of True or False based on the MQDQ syllable notation
            islong = [x in ("A", "T", "X") for x in re.findall(r"\d(.)", w.mqdq["sy"])]
            for syl, long in zip(w.syls, islong):
                if syl == "_":
                    # elided syllables don't exist
                    continue
                else:
                    if long:
                        d.text(
                            (xpos, ypos),
                            syl.translate(rhyme.DEMACRON).upper(),
                            font=fnt_small,
                            fill="black",
                            anchor="mm",
                        )
                        xpos += W
                    else:
                        d.text(
                            (xpos, ypos),
                            syl.translate(rhyme.DEMACRON).lower(),
                            font=fnt_small,
                            fill="black",
                            anchor="mm",
                        )
                        xpos += W

    im = _add_title(white, titles[idx])
    if fn:
        im.save(fn)
    return im
