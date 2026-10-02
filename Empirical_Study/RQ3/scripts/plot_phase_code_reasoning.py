import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RQ3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_DIR = os.path.join(RQ3_DIR, "data", "phase")
DEFAULT_OUTPUT = os.path.join(RQ3_DIR, "figures", "phase_code_reasoning_stacked.png")

# Pipeline phases to plot, in order. "general" is the catch-all bucket and is
# intentionally excluded.
PHASES = ["localization", "patch", "validation"]
PHASE_ABBR = {"localization": "L", "patch": "P", "validation": "V"}

# The node-count schema (claude/gpt/gemini/minimax exports) keys phases by the
# logical names above and counts distinct graph nodes.
NODE_SCHEMA = {
    "phase": {"localization": "localization", "validation": "validation",
              "patch": "patch"},
    "total": "distinct_nodes",
    "with_cr": "distinct_nodes_with_code_reasoning",
}
# The iCAT-Agent exports key phases by agent role and count turns. Map them
# onto the same logical loc/val/patch axis.
ICAT_SCHEMA = {
    "phase": {"localization": "localizer", "validation": "reproducer",
              "patch": "patch_editor"},
    "total": "turns",
    "with_cr": "turns_with_code_reasoning",
}

# Sets whose trajectories carry their own phase tags (Claude Code sub-agent
# roles, ExpeRepair pipeline phases; phase_code_reasoning_tagged.py) already
# use the logical names but count turns, like the iCAT exports.
TURN_SCHEMA = {
    "phase": {"localization": "localization", "validation": "validation",
              "patch": "patch"},
    "total": "turns",
    "with_cr": "turns_with_code_reasoning",
}

GREEN = "lightgreen"
CORAL = "lightcoral"

FILE_PREFIX = "phase_code_reasoning_"


def model_name(filename):
    """Strip the ``phase_code_reasoning_`` prefix and ``.json`` suffix."""
    base = os.path.basename(filename)
    if base.startswith(FILE_PREFIX):
        base = base[len(FILE_PREFIX):]
    if base.endswith(".json"):
        base = base[: -len(".json")]
    return base


def _detect_schema(phases):
    """Pick the schema whose phase keys / count fields are present.

    Returns the matching schema dict, or None if neither fits.
    """
    for schema in (NODE_SCHEMA, ICAT_SCHEMA, TURN_SCHEMA):
        if all(
            isinstance(
                phases.get(schema["phase"][ph], {}).get(schema["total"]),
                (int, float),
            )
            for ph in PHASES
        ):
            return schema
    return None


def load_models(input_dir):
    """Return [(model, {phase: (total, with_cr)})] sorted by model name.

    Handles both the node-count schema (distinct_nodes over loc/val/patch) and
    the iCAT-Agent turn-count schema (turns over localizer/reproducer/
    patch_editor), normalising both onto the same logical loc/val/patch axis.
    """
    names = sorted(
        n for n in os.listdir(input_dir)
        if n.startswith(FILE_PREFIX) and n.endswith(".json")
    )
    models = []
    skipped = []
    for name in names:
        with open(os.path.join(input_dir, name)) as f:
            data = json.load(f)
        phases = data.get("phases", {})
        schema = _detect_schema(phases)
        if schema is None:
            skipped.append(model_name(name))
            continue
        per_phase = {}
        for ph in PHASES:
            pd = phases.get(schema["phase"][ph], {})
            total = pd.get(schema["total"], 0)
            with_cr = pd.get(schema["with_cr"], 0)
            per_phase[ph] = (total, with_cr)
        models.append((model_name(name), per_phase))
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with unrecognised schema: "
              + ", ".join(skipped))
    # Same set order as the other figures: mini sbv, mini sbp, icat, openhands,
    # sonar, claudecode, experepair.
    order = ["mini_sbv", "mini_sbp", "icat_sbv", "icat_sbp", "openhands", "sonar",
             "claudecode", "experepair"]
    def _key(mp):
        n = mp[0]
        return (next((i for i, o in enumerate(order) if n.startswith(o)), len(order)), n)
    models.sort(key=_key)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help="directory of phase_code_reasoning_*.json files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output PNG path")
    args = parser.parse_args()

    models = load_models(args.input_dir)
    if not models:
        raise SystemExit(f"No {FILE_PREFIX}*.json files under {args.input_dir}")

    n_models = len(models)
    n_phases = len(PHASES)
    bar_w = 0.8 / n_phases          # bars fill ~80% of each model's slot
    group_gap = 1.0                 # one unit per model along the x-axis

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.6), 8))

    bar_centers = []                # for per-bar phase tick labels
    bar_labels = []
    group_centers = []              # for centered model labels

    for mi, (model, per_phase) in enumerate(models):
        x0 = mi * group_gap
        centers_this = []
        for pi, ph in enumerate(PHASES):
            total, with_cr = per_phase[ph]
            rest = max(total - with_cr, 0)
            # Offset each phase bar within the model's slot, centered on x0.
            xc = x0 + (pi - (n_phases - 1) / 2) * bar_w
            centers_this.append(xc)

            ax.bar(xc, with_cr, width=bar_w, color=GREEN,
                   edgecolor="black", linewidth=0.5)
            ax.bar(xc, rest, bottom=with_cr, width=bar_w, color=CORAL,
                   edgecolor="black", linewidth=0.5)

            # Annotate the code-reasoning percentage just above the green
            # segment (shown even when that segment is tiny).
            if total > 0:
                ax.text(xc, with_cr, f"{100 * with_cr / total:.1f}%",
                        ha="center", va="bottom", fontsize=28, rotation=90)

            bar_centers.append(xc)
            bar_labels.append(PHASE_ABBR.get(ph, ph))
        group_centers.append(sum(centers_this) / len(centers_this))

    ax.set_ylabel("#nodes", fontsize=30)
    ax.tick_params(axis="y", labelsize=28)

    # Phase abbreviations on the primary x-axis.
    ax.set_xticks(bar_centers)
    ax.set_xticklabels(bar_labels, fontsize=28)

    # Model names as a second row of labels below the phase ticks.
    ymin = ax.get_ylim()[0]
    span = ax.get_ylim()[1] - ymin
    for xc, (model, _) in zip(group_centers, models):
        ax.text(xc, ymin - 0.09 * span, model, ha="center", va="top",
                fontsize=18, rotation=25)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor="black",
                      label="with code reasoning"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CORAL, edgecolor="black",
                      label="without code reasoning"),
    ]
    ax.legend(handles=legend_handles, fontsize=30, loc="upper left")

    ax.margins(x=0.01)
    fig.subplots_adjust(bottom=0.32, left=0.07, right=0.99)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
