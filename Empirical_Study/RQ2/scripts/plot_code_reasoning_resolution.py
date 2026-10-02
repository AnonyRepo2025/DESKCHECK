
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RQ2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(RQ2_DIR, "data", "code_reasoning_resolution_allreasoning.csv")
OUT_PATH = os.path.join(RQ2_DIR, "figures", "code_reasoning_resolution_stacked.png")

GREEN = "lightgreen"
CORAL = "lightcoral"
HATCH = "//"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-scaffold", action="store_true",
                        help="sum reports per scaffold before plotting")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["n_instances", "n_resolved"]).reset_index(drop=True)
    df["report"] = df["report"].astype(str).str.strip()
    df["code_reasoning_group"] = df["code_reasoning_group"].astype(str).str.strip()

    # Drop the aggregate row; it dwarfs the per-report bars.
    df = df[df["report"] != "ALL_REPORTS_COMBINED"].reset_index(drop=True)

    out_path = OUT_PATH
    if args.by_scaffold:
        df["report"] = df["report"].str.split("_").str[0]
        df = (df.groupby(["report", "code_reasoning_group"], sort=False)
                [["n_instances", "n_resolved"]].sum().reset_index())
        out_path = OUT_PATH.replace(".png", "_by_scaffold.png")

    # Pivot so each report has both groups available in one row.
    reports = list(dict.fromkeys(df["report"]))
    by_key = {
        (r.report, r.code_reasoning_group): r for r in df.itertuples()
    }

    x = list(range(len(reports)))
    fig, ax = plt.subplots(figsize=(max(8, len(reports) * (2.2 if args.by_scaffold else 1.5)), 7))

    for xi, report in zip(x, reports):
        with_row = by_key.get((report, "with_code_reasoning"))
        without_row = by_key.get((report, "without_code_reasoning"))

        w_n = with_row.n_instances if with_row else 0
        w_res = with_row.n_resolved if with_row else 0
        wo_n = without_row.n_instances if without_row else 0
        wo_res = without_row.n_resolved if without_row else 0

        bottom = 0
        # With code reasoning (light green): resolved (hatched) then unresolved.
        ax.bar(xi, w_res, bottom=bottom, color=GREEN, edgecolor="black",
               hatch=HATCH, linewidth=0.5)
        bottom += w_res
        ax.bar(xi, w_n - w_res, bottom=bottom, color=GREEN, edgecolor="black",
               linewidth=0.5)
        bottom += w_n - w_res
        # Without code reasoning (lightcoral): resolved (hatched) then unresolved.
        ax.bar(xi, wo_res, bottom=bottom, color=CORAL, edgecolor="black",
               hatch=HATCH, linewidth=0.5)
        bottom += wo_res
        ax.bar(xi, wo_n - wo_res, bottom=bottom, color=CORAL, edgecolor="black",
               linewidth=0.5)
        bottom += wo_n - wo_res

        # Annotate each non-empty segment with its count.
        total = w_n + wo_n
        segs = [
            (w_res / 2, w_res),
            (w_res + (w_n - w_res) / 2, w_n - w_res),
            (w_n + wo_res / 2, wo_res),
            (w_n + wo_res + (wo_n - wo_res) / 2, wo_n - wo_res),
        ]
        for ypos, val in segs:
            if val > 0.02 * max(total, 1):
                ax.text(xi, ypos, str(int(val)), rotation=0, ha="center",
                        va="center", color="black", fontsize=17)

    ax.set_ylabel("#instances", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xticks(x)
    if args.by_scaffold:
        ax.set_xticklabels(reports, fontsize=16)
    else:
        ax.set_xticklabels(reports, rotation=30, ha="right", fontsize=10)

    # Legend: colours for the code-reasoning split, hatch for resolution.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor="black",
                      label="with code reasoning"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CORAL, edgecolor="black",
                      label="without code reasoning"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      hatch=HATCH, label="resolved"),
    ]
    ax.legend(handles=legend_handles, fontsize=16)

    fig.subplots_adjust(bottom=0.12 if args.by_scaffold else 0.32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
