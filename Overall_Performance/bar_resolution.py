#!/usr/bin/env python3
"""Grouped bar charts per model: baseline vs. pipeline instances resolved by difficulty and language.

Reads data/<model>.csv (same files as overall_resolution.py). Bar height is the resolved count;
each bar's label is that count as a share of its group (the groups differ in size).
Every model's chart shares one y axis (0-300), so the charts are comparable side by side.
Colours: green = baseline, blue = pipeline (no legend on the chart).

Usage
    python3 bar_resolution.py [--out-dir DIR] [--y-max 300]

Writes <model>_resolution_by_difficulty.png and <model>_resolution_by_language.png for each
model to --out-dir (default bars/ next to this script).
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_OUT = HERE / "bars"
MODELS = [
    ("MiniMax-M3", "MiniMax M3", "minimax_m3.csv"),
    ("GPT-5.6-Luna", "GPT-5.6 Luna", "gpt56_luna.csv"),
    ("ccpipe-Haiku", "Claude Haiku (Claude Code)", "claude_haiku.csv"),
]
GROUP_ORDER = {
    "difficulty": [("easy", "Easy"), ("medium", "Medium"), ("difficult", "Difficult")],
    "language": [("python", "Python"), ("go", "Go"), ("js", "JavaScript"), ("ts", "TypeScript")],
}
FIG_WIDTH = {"difficulty": None, "language": 5.4}   # language labels are long
# difficulty charts: fonts x1.2 on a fixed 672x510 px canvas (the size the default layout gives),
# so the bigger text shrinks the plot area instead of growing the image
FONT_SCALE = {"difficulty": 1.2, "language": 1.0}
# numbers (bar percentages, y-axis ticks) on the difficulty charts get their own, larger scale
NUMBER_SCALE = {"difficulty": 1.85, "language": 1.0}       # bar percentages
TICK_SCALE = {"difficulty": 1.4, "language": 1.0}        # y-axis numbers
CANVAS_PX = {"difficulty": (672, 510), "language": None}
DPI = 170

BASELINE_COLOR = "#1a9e77"
PIPELINE_COLOR = "#2a78d6"
# charts drawn with an orange baseline instead (same orange as the reasoning_intervention figures)
ORANGE_BASELINE = "#eb6834"
ORANGE_BASELINE_CHARTS = {("GPT-5.6-Luna", "difficulty"), ("MiniMax-M3", "language"),
                          ("MiniMax-M3", "difficulty")}
FILL_ALPHA = 0.55
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d4"
FONT_TICK, FONT_LABEL, FONT_VALUE, FONT_AXIS_TITLE = 14, 14, 12, 19


def load(name):
    with open(DATA / name, newline="") as f:
        return [r for r in csv.DictReader(f)
                if r["baseline"] in ("PASS", "FAIL") and r["pipeline"] in ("PASS", "FAIL")]


def by_group(rows, group):
    out = []
    for key, label in GROUP_ORDER[group]:
        grp = [r for r in rows if r[group] == key]
        if not grp:
            continue
        row = {"name": label, "n": len(grp)}
        for side in ("baseline", "pipeline"):
            k = sum(r[side] == "PASS" for r in grp)
            row[f"{side}_resolved"], row[f"{side}_rate"] = k, k / len(grp)
        out.append(row)
    return out


def draw(rows, out_path, y_max, y_step=100, fig_width=None, font_scale=1.0, canvas_px=None,
         number_scale=1.0, tick_scale=1.0, axes_box=None, baseline_color=BASELINE_COLOR):
    """With canvas_px, returns the plot area (figure fractions); passing that back as axes_box
    gives every chart the same plot area, whichever chart's labels need the most room."""
    # the axis title is already the largest text and spans nearly the full height, so it keeps
    # its size
    font_label = FONT_LABEL * font_scale
    font_tick, font_value = FONT_TICK * tick_scale, FONT_VALUE * number_scale
    figsize = ((canvas_px[0] / DPI, canvas_px[1] / DPI) if canvas_px
               else (fig_width or (0.72 * len(rows) + 1.7), 2.9))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    width, gap, step = 0.24, 0.012, 0.58
    xs = [i * step for i in range(len(rows))]
    top = max(max(r["baseline_resolved"], r["pipeline_resolved"]) for r in rows)
    for key, color, side in (("baseline", baseline_color, -1), ("pipeline", PIPELINE_COLOR, 1)):
        offset = side * (width / 2 + gap)
        counts = [r[f"{key}_resolved"] for r in rows]
        ax.bar([x + offset for x in xs], counts, width, color=color, alpha=FILL_ALPHA,
               linewidth=0, zorder=3)
        # rotated labels, anchored at the bar top so they grow upward
        for x, r, c in zip(xs, rows, counts):
            ax.text(x + offset, c + 0.02 * top, f"{100 * r[f'{key}_rate']:.1f}%", ha="left",
                    va="center", rotation=90, rotation_mode="anchor",
                    fontsize=font_value, color=TEXT_PRIMARY, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"] for r in rows], fontsize=font_label, color=TEXT_PRIMARY)
    ax.set_ylabel("Instances resolved", fontsize=FONT_AXIS_TITLE, color="#000000")
    ax.set_ylim(0, y_max)
    ax.set_yticks(list(range(0, y_max + 1, y_step)))
    # on a fixed canvas the outer margins give way, so the bigger group labels do not collide
    edge = 0.42 if canvas_px else 0.62
    ax.set_xlim(xs[0] - step * edge, xs[-1] + step * edge)
    ax.tick_params(axis="y", labelsize=font_tick, labelcolor=TEXT_SECONDARY, length=4,
                   color=TEXT_SECONDARY)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color(TEXT_SECONDARY)

    if canvas_px:
        fig.tight_layout(pad=0.2 * 72 / font_label)
        if axes_box:
            ax.set_position(axes_box)
        box = ax.get_position()
        if out_path is not None:
            fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
        plt.close(fig)
        return box
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--y-max", type=int, default=300, help="shared axis top (default 300)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    style = lambda g: dict(fig_width=FIG_WIDTH[g], font_scale=FONT_SCALE[g], canvas_px=CANVAS_PX[g],
                           number_scale=NUMBER_SCALE[g], tick_scale=TICK_SCALE[g])
    data = {stem: load(fname) for stem, _, fname in MODELS}
    # fixed-canvas groups: lay every model's chart out once, then draw all of them in the plot
    # area that fits them all, so the charts stay the same geometry side by side
    shared = {}
    for group in GROUP_ORDER:
        if CANVAS_PX[group]:
            boxes = [draw(by_group(data[stem], group), None, args.y_max, **style(group))
                     for stem, _, _ in MODELS]
            x0, y0 = max(b.x0 for b in boxes), max(b.y0 for b in boxes)
            x1, y1 = min(b.x1 for b in boxes), min(b.y1 for b in boxes)
            shared[group] = (x0, y0, x1 - x0, y1 - y0)

    for stem, model, fname in MODELS:
        print(f"== {model}")
        for group in GROUP_ORDER:
            rows = by_group(data[stem], group)
            # the model name carries dots (GPT-5.6), so build the filename rather than use with_suffix
            draw(rows, args.out_dir / f"{stem}_resolution_by_{group}.png", args.y_max,
                 axes_box=shared.get(group), **style(group),
                 baseline_color=ORANGE_BASELINE if (stem, group) in ORANGE_BASELINE_CHARTS
                 else BASELINE_COLOR)
            for r in rows:
                print(f"  {r['name']:<10} n={r['n']:>3}  baseline {r['baseline_resolved']:>3} "
                      f"({r['baseline_rate']:6.1%})  pipeline {r['pipeline_resolved']:>3} "
                      f"({r['pipeline_rate']:6.1%})")
    print(f"wrote *_resolution_by_{{difficulty,language}}.png to {args.out_dir}")


if __name__ == "__main__":
    main()
