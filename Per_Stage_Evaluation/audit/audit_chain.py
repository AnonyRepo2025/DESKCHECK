#!/usr/bin/env python3
"""Three-stage Sankey of the code-audit chain: violation -> code edit -> resolution.

Each instance flows through the pipeline's spec-audit step and out to the final verdict:
    violation   (check mark)  the spec audit recorded a violation      / (cross) it recorded none
    code edit   Changed       an audit-fix round edited the patch       / Unchanged
    resolution  PASS / FAIL   pipeline outcome on SWE-bench Pro (all FAIL_TO_PASS and PASS_TO_PASS tests pass)
Ribbons take their destination node's colour. The few instances whose patch changed without a
recorded violation are drawn as violation -> changed (the data files keep the recorded flags).

Two figures per model:
    audit_chain.png         all 731 instances
    audit_chain_rescue.png  only instances the baseline agent FAILED (PASS = rescued by the pipeline)

Data: data/<model>.csv, one row per instance with columns
    instance_id, language, difficulty, baseline (PASS/FAIL), pipeline (PASS/FAIL),
    audit_violation, audit_change ("Y" / "Y*" = set, "Y*" being hand-verified; blank = not set)

Usage:
    python3 audit_chain.py                    # writes into ./figures/<model>/
    python3 audit_chain.py --out-dir DIR      # writes into DIR/<model>/
    python3 audit_chain.py --out MODEL=DIR    # explicit directory per model (repeatable)
Requires matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
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

# Figure settings of the paper: 75%-width panel, 1.3x fonts, symbol-marked first column, no headers
WIDTH_SCALE, FONT_SCALE = 0.75, 1.3
PANEL_W, PANEL_H = 9.5, 6.4

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
FONT_NODE, FONT_RIBBON = 15, 12


@dataclass
class Stage:
    field: str
    order: "list[str]"
    labels: "dict[str, str]"
    colors: "dict[str, str]"
    label_side: str = "right"


STAGES = [
    Stage("audit_violation", ["Y", ""], {"Y": "Violation", "": "No violation"},
          {"Y": "#eda100", "": "#8a8983"}, label_side="left"),
    Stage("audit_change", ["Y", ""], {"Y": "Changed", "": "Unchanged"},
          {"Y": "#2a78d6", "": "#8a8983"}),
    Stage("pipeline", ["PASS", "FAIL"], {"PASS": "PASS", "FAIL": "FAIL"},
          {"PASS": "#0ca30c", "FAIL": "#d03b3b"}),
]
SRC_MARK = {"Y": ("✔", "#0ca30c"), "": ("✘", "#d03b3b")}


def flag(v: str) -> str:
    return "Y" if v.strip().upper().startswith("Y") else ""


def load(path: Path) -> "list[dict]":
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        assert r["baseline"] in ("PASS", "FAIL") and r["pipeline"] in ("PASS", "FAIL"), r
        r["audit_violation"], r["audit_change"] = flag(r["audit_violation"]), flag(r["audit_change"])
    assert len({r["instance_id"] for r in rows}) == len(rows), f"duplicate ids in {path}"
    return rows


def for_figure(rows: "list[dict]") -> "list[dict]":
    """Draw 'no violation -> changed' instances as 'violation -> changed'."""
    return [dict(r, audit_violation="Y") if r["audit_violation"] == "" and r["audit_change"] == "Y" else r
            for r in rows]


def link_counts(rows, a: Stage, b: Stage) -> "dict[tuple[str, str], int]":
    c = Counter((r[a.field], r[b.field]) for r in rows)
    return {(x, y): c.get((x, y), 0) for x in a.order for y in b.order}


def node_totals(rows, st: Stage) -> "dict[str, int]":
    c = Counter(r[st.field] for r in rows)
    return {k: c.get(k, 0) for k in st.order}


def print_links(title: str, rows: "list[dict]") -> None:
    print(f"  {title}: {len(rows)} instances")
    for i in range(len(STAGES) - 1):
        a, b = STAGES[i], STAGES[i + 1]
        counts, tot = link_counts(rows, a, b), node_totals(rows, a)
        for x in a.order:
            if tot[x]:
                parts = [f"{b.labels[y]} {counts[(x, y)]} ({counts[(x, y)] / tot[x]:.0%})"
                         for y in b.order if counts[(x, y)]]
                print(f"    {a.labels[x]:<13} n={tot[x]:<4} -> " + ", ".join(parts))


def _ribbon(x0, y0a, y0b, x1, y1a, y1b) -> MplPath:
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b),
             (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def draw_chain(ax, rows: "list[dict]") -> None:
    """Narrow-panel layout: middle-column labels above/below their bar, ribbon labels mid-link on
    two lines outlined in their destination colour, side margins sized from the label text."""
    side_texts: "dict[str, list]" = {"left": [], "right": []}
    font_node, font_ribbon = FONT_NODE * FONT_SCALE, FONT_RIBBON * FONT_SCALE
    total = len(rows)
    gap = 0.05 * total
    totals = [node_totals(rows, st) for st in STAGES]
    height = total + gap * (max(sum(1 for v in t.values() if v) for t in totals) - 1)

    def layout(st: Stage, tot):
        pos, y = {}, height
        for k in st.order:
            if tot[k] == 0:
                continue
            pos[k] = (y - tot[k], y)
            y -= tot[k] + gap
        return pos

    positions = [layout(st, t) for st, t in zip(STAGES, totals)]
    n = len(STAGES)
    w = 0.045
    xs = [i / (n - 1) * (1 - w) for i in range(n)]

    for i, (st, pos, tot) in enumerate(zip(STAGES, positions, totals)):
        for k, (b, t) in pos.items():
            ax.add_patch(Rectangle((xs[i], b), w, t - b, facecolor=st.colors[k], edgecolor="none", zorder=4))
            if i == 0:          # "<symbol> <count>" on one line
                glyph, colour = SRC_MARK[k]
                count = ax.text(xs[i] - 0.02, (b + t) / 2, f"{tot[k]:,}", ha="right", va="center",
                                fontsize=font_node, color=TEXT_PRIMARY, zorder=6)
                side_texts["left"].append(count)
                fig = ax.figure
                count_pt = count.get_window_extent(renderer=fig.canvas.get_renderer()).width * 72 / fig.dpi
                side_texts["left"].append(ax.annotate(
                    glyph, (xs[i] - 0.02, (b + t) / 2), xytext=(-count_pt - font_node * 0.3, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=font_node * 1.3,
                    color=colour, fontweight="bold", zorder=6))
            elif i < n - 1:     # middle column: first node labelled above its bar, the others below
                above = k == next(iter(pos))
                ax.text(xs[i] + w / 2, t + gap * 0.25 if above else b - gap * 0.25,
                        f"{st.labels[k]} {tot[k]:,}", ha="center", va="bottom" if above else "top",
                        fontsize=font_node, color=TEXT_PRIMARY, zorder=6)
            else:
                side_texts["right"].append(ax.text(
                    xs[i] + w + 0.02, (b + t) / 2, f"{st.labels[k]}\n{tot[k]:,}", ha="left", va="center",
                    fontsize=font_node, color=TEXT_PRIMARY, linespacing=1.1, zorder=6, bbox=None))

    for i in range(n - 1):
        a, b = STAGES[i], STAGES[i + 1]
        counts = link_counts(rows, a, b)
        out_cur = {k: positions[i][k][1] for k in positions[i]}
        in_cur = {k: positions[i + 1][k][1] for k in positions[i + 1]}
        labels = []
        for x in a.order:
            for y in b.order:
                v = counts.get((x, y), 0)
                if v == 0:
                    continue
                ya_t, ya_b = out_cur[x], out_cur[x] - v
                yb_t, yb_b = in_cur[y], in_cur[y] - v
                out_cur[x], in_cur[y] = ya_b, yb_b
                ax.add_patch(PathPatch(_ribbon(xs[i] + w, ya_t, ya_b, xs[i + 1], yb_t, yb_b),
                                       facecolor=b.colors[y], edgecolor=SURFACE, linewidth=0.6,
                                       alpha=0.42 if x == y else 0.72, zorder=2))
                y_at = ((ya_t + ya_b) / 2) * 0.5 + ((yb_t + yb_b) / 2) * 0.5
                labels.append([y_at, f"{v:,}\n({v / totals[i][x]:.0%})", b.colors[y]])
        labels.sort(key=lambda t: -t[0])
        min_dy = 0.10 * height * FONT_SCALE
        for j in range(1, len(labels)):
            if labels[j - 1][0] - labels[j][0] < min_dy:
                labels[j][0] = labels[j - 1][0] - min_dy
        xm = (xs[i] + w) + 0.5 * (xs[i + 1] - xs[i] - w)
        for y, txt, edge in labels:
            ax.text(xm, y, txt, ha="center", va="center", fontsize=font_ribbon, color=TEXT_SECONDARY,
                    zorder=5, linespacing=1.0,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor=edge,
                              linewidth=1.4, alpha=0.85))

    below = gap * 1.6 * FONT_SCALE          # room for the middle column's bottom label
    fig = ax.figure
    r = fig.canvas.get_renderer()

    def need(side):     # widest label on that side, in inches
        return max((tx.get_window_extent(renderer=r).width for tx in side_texts[side]), default=0) / fig.dpi

    left_in = need("left") + font_node * 1.6 / 72 + 0.08
    right_in = need("right") + 0.08
    axes_in = ax.get_position().width * fig.get_figwidth()
    unit = max(axes_in - left_in - right_in, 0.3 * axes_in)
    ax.set_xlim(-left_in / unit, 1 + right_in / unit)
    ax.set_ylim(-below - gap * 1.0, height + 1.6 * FONT_SCALE * gap)
    ax.set_axis_off()


def render(rows: "list[dict]", out_png: Path) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(PANEL_W * WIDTH_SCALE, PANEL_H), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0][0]
    ax.set_facecolor(SURFACE)
    draw_chain(ax, rows)
    fig.tight_layout(w_pad=3.0, h_pad=2.0)
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
        print(display)
        for stem, subset, title in (("audit_chain", rows, "all instances"),
                                    ("audit_chain_rescue", [r for r in rows if r["baseline"] == "FAIL"],
                                     "baseline-failed instances")):
            fig_rows = for_figure(subset)
            print_links(f"{title} (as drawn)", fig_rows)
            render(fig_rows, out / f"{stem}.png")
        print(f"  wrote {out}/audit_chain.png, audit_chain_rescue.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
