import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RQ3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(RQ3_DIR, "data", "difficulty.csv")
OUT_PATH = os.path.join(RQ3_DIR, "figures", "difficulty_stacked.png")

DIFFICULTIES = ["easy", "medium", "difficult"]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    rows = load_rows(CSV_PATH)

    # Detect whether scaffold is needed to disambiguate model-benchmark labels.
    base_keys = [(r["model"], r["benchmark"]) for r in rows]
    needs_scaffold = len(set(base_keys)) != len(base_keys)

    labels = []
    for r in rows:
        label = f"{r['model']}\n{r['benchmark']}"
        if needs_scaffold:
            label += f"\n({r['scaffold']})"
        labels.append(label)

    n_groups = len(rows)
    n_diff = len(DIFFICULTIES)
    bar_width = 0.25
    group_gap = 0.10

    # x position for each (group, difficulty) bar.
    group_span = n_diff * bar_width
    group_starts = np.arange(n_groups) * (group_span + group_gap)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.2), 6))

    PCT_FONTSIZE = 22
    LETTER_FONTSIZE = 22
    ymax = max(int(r[f"n_{d}"]) for r in rows for d in DIFFICULTIES)
    ax.set_ylim(0, ymax * 1.15)
    # (x, total, code, pct) per bar; labels are placed after measuring text size.
    bars = []

    for di, diff in enumerate(DIFFICULTIES):
        xs = group_starts + di * bar_width + bar_width / 2
        totals = np.array([int(r[f"n_{diff}"]) for r in rows], dtype=float)
        code = np.array([int(r[f"code_reasoning_{diff}"]) for r in rows], dtype=float)
        rest = totals - code

        ax.bar(xs, code, width=bar_width, color="lightgreen",
               edgecolor="black", linewidth=0.6,
               label="with code reasoning" if di == 0 else None)
        ax.bar(xs, rest, width=bar_width, bottom=code, color="lightcoral",
               edgecolor="black", linewidth=0.6,
               label="without code reasoning" if di == 0 else None)

        for x, total, c in zip(xs, totals, code):
            bars.append((x, total, c, diff[0].upper()))

    ax.set_xticks(group_starts + group_span / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", labelsize=22)
    # ax.set_yticks(ax.get_yticks(), fontsize=16)
    ax.set_ylabel("# instances", fontsize=22)
    # ax.set_title("Code reasoning by difficulty (E=easy, M=medium, D=difficult)", fontsize=12)
    ax.legend(fontsize=22)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    # Freeze the layout first: label heights are measured in data units, which
    # would shift if tight_layout resized the axes afterwards.
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_top(t):
        bb = t.get_window_extent(renderer=renderer)
        return inv.transform((bb.x0, bb.y1))[1]

    pad = 0.01 * ymax
    for x, total, c, letter in bars:
        top = total
        if total > 0:
            pct = f"{c / total * 100:.0f}%"
            t = ax.text(x, c / 2, pct, rotation=90, ha="center", va="center",
                        fontsize=PCT_FONTSIZE, color="black")
            bb = t.get_window_extent(renderer=renderer)
            height = inv.transform((0, bb.y1))[1] - inv.transform((0, bb.y0))[1]
            if height > c * 0.95:
                # Does not fit in the green segment: start the label just inside
                # the bottom of the bar and let it extend upward.
                t.set_position((x, max(0.1 * c, pad)))
                t.set_va("bottom")
                top = max(total, text_top(t))
        # Difficulty letter above the bar and above any label that sticks out.
        ax.text(x, top + pad, letter, ha="center", va="bottom",
                fontsize=LETTER_FONTSIZE, color="black")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
