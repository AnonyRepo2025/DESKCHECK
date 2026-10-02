#!/usr/bin/env python3
"""Sankey: does a post-validation edit lead to a resolved patch?

Each instance flows from its post-validation flag to the pipeline's final resolution:
    Edited      (check mark)  a gate after the spec audit (validate / regression fix) edited the patch
    Not edited  (cross)       no post-audit gate changed it
    -> PASS / FAIL             pipeline outcome on SWE-bench Pro (all FAIL_TO_PASS and PASS_TO_PASS tests pass)

Two figures per model:
    postvalidation_resolution.png         all 731 instances
    postvalidation_resolution_rescue.png  only instances the baseline agent FAILED (PASS = rescued by the pipeline)

Data: data/<model>.csv, one row per instance with columns
    instance_id, language, difficulty, baseline (PASS/FAIL), pipeline (PASS/FAIL),
    post_validation_edit ("Y" or "Y*" = edited, "Y*" being hand-verified; blank = not edited)

Usage:
    python3 postvalidation_resolution.py                    # writes into ./figures/<model>/
    python3 postvalidation_resolution.py --out-dir DIR      # writes into DIR/<model>/
    python3 postvalidation_resolution.py --out MODEL=DIR    # explicit directory per model (repeatable)
Requires matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS = {  # key -> (data file, display name)
    "minimax_m3": ("minimax_m3.csv", "MiniMax-M3"),
    "gpt56_luna": ("gpt56_luna.csv", "GPT-5.6 Luna"),
    "claude_haiku": ("claude_haiku.csv", "Claude Haiku 4.5 (Claude Code)"),
}

# Figure settings of the paper: half-width, 60%-height panel, 1.2x labels, symbol-marked sources, no headers
WIDTH_SCALE, HEIGHT_SCALE, FONT_SCALE = 0.5, 0.6, 1.2
PANEL_W, PANEL_H = 6.4, 5.6

SRC = ["Y", ""]
SRC_LABEL = {"Y": "Edited", "": "Not edited"}
SRC_COLOR = {"Y": "#2a78d6", "": "#8a8983"}
DST = ["PASS", "FAIL"]
DST_LABEL = {"PASS": "PASS", "FAIL": "FAIL"}
DST_LABEL_RESCUE = DST_LABEL
DST_COLOR = {"PASS": "#0ca30c", "FAIL": "#d03b3b"}
SRC_MARK = {"Y": ("✔", DST_COLOR["PASS"]), "": ("✘", DST_COLOR["FAIL"])}

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
FONT_NODE, FONT_RIBBON = 16, 14


def load(path: Path) -> "list[dict]":
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        assert r["baseline"] in ("PASS", "FAIL") and r["pipeline"] in ("PASS", "FAIL"), r
        r["edited"] = "Y" if r["post_validation_edit"].strip().upper().startswith("Y") else ""
    assert len({r["instance_id"] for r in rows}) == len(rows), f"duplicate ids in {path}"
    return rows


def flow_matrix(rows: "list[dict]") -> "dict[tuple[str, str], int]":
    m = Counter((r["edited"], r["pipeline"]) for r in rows)
    return {(a, b): m.get((a, b), 0) for a in SRC for b in DST}


def print_matrix(title: str, m: "dict[tuple[str, str], int]") -> None:
    total = sum(m.values())
    print(f"  {title}: {total} instances")
    for a in SRC:
        n = m[(a, "PASS")] + m[(a, "FAIL")]
        rate = f"{m[(a, 'PASS')] / n:.1%}" if n else "-"
        print(f"    {SRC_LABEL[a]:<12} pass {m[(a, 'PASS')]:>4}  fail {m[(a, 'FAIL')]:>4}  total {n:>4}  pass-rate {rate}")


def _ribbon(x0, y0a, y0b, x1, y1a, y1b) -> MplPath:
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b),
             (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def draw_sankey(ax, m: "dict[tuple[str, str], int]", dst_label: "dict[str, str]") -> None:
    """Two-column Sankey, ribbons coloured by outcome; compact layout (count over share mid-ribbon)."""
    fs = min(1.0, 0.5 + 0.5 * WIDTH_SCALE) * FONT_SCALE
    font_node, font_ribbon = FONT_NODE * fs, FONT_RIBBON * fs
    total = sum(m.values())
    left = {a: sum(m[(a, b)] for b in DST) for a in SRC}
    right = {b: sum(m[(a, b)] for a in SRC) for b in DST}
    gap = 0.04 * total
    n_gap = max(len([a for a in SRC if left[a]]), len([b for b in DST if right[b]])) - 1
    height = total + gap * max(n_gap, 0)

    def layout(order, counts):
        pos, y = {}, height
        for s in order:
            if counts[s] == 0:
                continue
            pos[s] = (y - counts[s], y)
            y -= counts[s] + gap
        return pos

    lpos, rpos = layout(SRC, left), layout(DST, right)
    x0, x1, w = 0.0, 1.0, 0.06 / WIDTH_SCALE

    for s, (b, t) in lpos.items():
        ax.add_patch(Rectangle((x0, b), w, t - b, facecolor=SRC_COLOR[s], edgecolor="none"))
        glyph, colour = SRC_MARK[s]            # "<symbol> <count>" on one line
        count = ax.text(x0 - 0.03, (b + t) / 2, f"{left[s]:,}", ha="right", va="center",
                        fontsize=font_node, color=TEXT_PRIMARY)
        fig = ax.figure
        count_pt = count.get_window_extent(renderer=fig.canvas.get_renderer()).width * 72 / fig.dpi
        ax.annotate(glyph, (x0 - 0.03, (b + t) / 2), xytext=(-count_pt - font_node * 0.3, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=font_node * 1.3, color=colour, fontweight="bold")
    for s, (b, t) in rpos.items():
        ax.add_patch(Rectangle((x1 - w, b), w, t - b, facecolor=DST_COLOR[s], edgecolor="none"))
        ax.text(x1 + 0.03, (b + t) / 2, f"{dst_label[s]}\n{right[s]:,}", ha="left", va="center",
                fontsize=font_node, color=TEXT_PRIMARY, linespacing=1.1)

    l_cur = {s: lpos[s][1] for s in lpos}
    r_cur = {s: rpos[s][1] for s in rpos}
    labels = []
    for a in SRC:
        for b in DST:
            v = m[(a, b)]
            if v == 0:
                continue
            ya_t, ya_b = l_cur[a], l_cur[a] - v
            yb_t, yb_b = r_cur[b], r_cur[b] - v
            l_cur[a], r_cur[b] = ya_b, yb_b
            ax.add_patch(PathPatch(_ribbon(x0 + w, ya_t, ya_b, x1 - w, yb_t, yb_b),
                                   facecolor=DST_COLOR[b], edgecolor=SURFACE, linewidth=0.6,
                                   alpha=0.45 if a == "" else 0.75, zorder=2 if a else 1))
            # source side, where one node's ribbons have not crossed the other's yet
            labels.append([(ya_t + ya_b) / 2 * 0.75 + (yb_t + yb_b) / 2 * 0.25,
                           f"{v:,}\n({v / left[a]:.0%})", DST_COLOR[b]])

    labels.sort(key=lambda t: -t[0])
    min_dy = 0.09 * height / HEIGHT_SCALE * FONT_SCALE
    for i in range(1, len(labels)):
        if labels[i - 1][0] - labels[i][0] < min_dy:
            labels[i][0] = labels[i - 1][0] - min_dy
    for y, txt, edge in labels:
        box = dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor=edge, linewidth=1.4, alpha=0.85)
        ax.text(0.5, y, txt, ha="center", va="center", fontsize=font_ribbon, linespacing=1.0,
                color=TEXT_SECONDARY, bbox=box, zorder=5)

    xpad = 0.45 / WIDTH_SCALE * FONT_SCALE
    ax.set_xlim(-xpad, 1 + xpad)
    ax.set_ylim(-gap * 1.0, height + gap * 1.0)
    ax.set_axis_off()


def render(m: "dict[tuple[str, str], int]", out_png: Path, dst_label: "dict[str, str]") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(PANEL_W * WIDTH_SCALE, PANEL_H * HEIGHT_SCALE), squeeze=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    draw_sankey(ax, m, dst_label)
    fig.tight_layout(w_pad=4.0)
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures",
                    help="root output directory; figures go to <out-dir>/<model>/ (default: ./figures)")
    ap.add_argument("--out", action="append", default=[], metavar="MODEL=DIR",
                    help=f"explicit output directory for one model ({', '.join(MODELS)}); repeatable")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    explicit = dict(o.split("=", 1) for o in args.out)
    unknown = set(explicit) - set(MODELS)
    if unknown:
        ap.error(f"unknown model(s) in --out: {sorted(unknown)}")

    for key in args.models:
        fname, display = MODELS[key]
        rows = load(DATA / fname)
        out = Path(explicit.get(key, args.out_dir / key))
        out.mkdir(parents=True, exist_ok=True)
        rescue = [r for r in rows if r["baseline"] == "FAIL"]
        print(f"{display}")
        for stem, subset, title, labels in (
                ("postvalidation_resolution", rows, "all instances", DST_LABEL),
                ("postvalidation_resolution_rescue", rescue, "baseline-failed instances", DST_LABEL_RESCUE)):
            m = flow_matrix(subset)
            print_matrix(title, m)
            render(m, out / f"{stem}.png", labels)
        print(f"  wrote {out}/postvalidation_resolution.png, postvalidation_resolution_rescue.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
