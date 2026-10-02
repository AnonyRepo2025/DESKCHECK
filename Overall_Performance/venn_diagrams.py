#!/usr/bin/env python3
"""Area-proportional Venn diagrams of baseline vs. pipeline resolved instances, one per model.

Reads data/<model>.csv (same files as overall_resolution.py) and treats resolution as two sets
over the same 731 instances:

    baseline only   the pipeline REGRESSED an instance the baseline had solved
    both            solved by either approach
    pipeline only   the pipeline RESCUED an instance the baseline failed
    neither         not drawn (a Venn has no region for it)

Needs matplotlib only (circles are drawn directly, no matplotlib_venn).

Usage
    python3 venn_diagrams.py [--out-dir DIR]

Writes venn_<m>.png for each model <m> in {minimax, luna, haiku} to --out-dir
(default venn/ next to this script).
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = Path(__file__).resolve().parent / "venn"
MODELS = [
    ("minimax", "MiniMax M3", "minimax_m3.csv"),
    ("luna", "GPT-5.6 Luna", "gpt56_luna.csv"),
    ("haiku", "Claude Haiku (Claude Code)", "claude_haiku.csv"),
]

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BASELINE_COLOR = "#eb6834"      # orange
PIPELINE_COLOR = "#2a78d6"      # blue
FONT_REGION = 44


def load(name):
    with open(DATA / name, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["baseline"] in ("PASS", "FAIL") and r["pipeline"] in ("PASS", "FAIL")]


def regions(rows):
    """(baseline only, both, pipeline only, neither)."""
    b, p = [r["baseline"] == "PASS" for r in rows], [r["pipeline"] == "PASS" for r in rows]
    pairs = list(zip(b, p))
    return (pairs.count((True, False)), pairs.count((True, True)),
            pairs.count((False, True)), pairs.count((False, False)))


# ---------------------------------------------------------------- geometry

def lens_area(r1, r2, d):
    """Intersection area of two circles with radii r1, r2 and centre distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    a3 = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - a3


def centre_distance(r1, r2, overlap):
    """Distance at which the lens area equals `overlap` (bisection; area falls as d grows)."""
    lo, hi = abs(r1 - r2), r1 + r2
    if overlap <= 0:
        return hi
    if overlap >= math.pi * min(r1, r2) ** 2:
        return lo
    for _ in range(100):
        mid = (lo + hi) / 2
        if lens_area(r1, r2, mid) > overlap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def draw_venn(ax, rows, colors=(BASELINE_COLOR, PIPELINE_COLOR)):
    b_only, both, p_only, _ = regions(rows)
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_aspect("equal")
    if b_only + both + p_only == 0:
        ax.text(0.5, 0.5, "no instances resolved", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_SECONDARY)
        return

    # radii so that circle area is proportional to the set size (unit: area 1 per instance)
    ra = math.sqrt((b_only + both) / math.pi)
    rb = math.sqrt((p_only + both) / math.pi)
    d = centre_distance(ra, rb, both)
    xa, xb = -d / 2, d / 2
    for x, r, c in ((xa, ra, colors[0]), (xb, rb, colors[1])):
        if r > 0:
            ax.add_patch(Circle((x, 0), r, facecolor=c, edgecolor="none", alpha=0.55))

    scale = max(ra, rb)
    left_edge, right_edge = xa - ra, xb + rb
    lens_l, lens_r = max(xb - rb, left_edge), min(xa + ra, right_edge)
    min_w = 0.45 * scale  # narrower than this and the number goes outside the circle

    def put(x, count):
        ax.text(x, 0, f"{count}", ha="center", va="center", fontsize=FONT_REGION, color=TEXT_PRIMARY)

    if both:
        put((lens_l + lens_r) / 2, both)
    if b_only:
        w = lens_l - left_edge
        put((left_edge + lens_l) / 2 if w >= min_w else left_edge - 0.35 * scale, b_only)
    if p_only:
        w = right_edge - lens_r
        put((lens_r + right_edge) / 2 if w >= min_w else right_edge + 0.35 * scale, p_only)

    pad = 0.75 * scale
    ax.set_xlim(left_edge - pad, right_edge + pad)
    ax.set_ylim(-scale * 1.1, scale * 1.1)


def render(rows, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    fig.patch.set_facecolor(SURFACE)
    draw_venn(ax, rows)
    fig.savefig(out_path, dpi=170, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, model, fname in MODELS:
        rows = load(fname)
        render(rows, args.out_dir / f"venn_{key}.png")
        bo, bt, po, ne = regions(rows)
        print(f"{model:<28} baseline-only {bo:>3}  both {bt:>3}  pipeline-only {po:>3}  neither {ne:>3}")
    print(f"wrote venn_{{minimax,luna,haiku}}.png to {args.out_dir}")


if __name__ == "__main__":
    main()
