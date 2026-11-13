"""Turn Inspect logs into a per-episode table, a summary table and the payoff-response chart.

    python analysis/summarize.py results/mvp
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

# Within each family the first model is the more capable tier.
FAMILIES = {
    "Anthropic": [
        ("openrouter/anthropic/claude-fable-5.2", "Claude Fable 5.2"),
        ("openrouter/anthropic/claude-haiku-5.5", "Claude Haiku 5.5"),
    ],
    "Google": [
        ("openrouter/google/gemini-4-carbon", "Gemini 4 Carbon"),
        ("openrouter/google/gemini-4-hydrogen", "Gemini 4 Hydrogen"),
    ],
}
# Chart tokens (light surface). Colour marks the tier and is the same in every panel.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TIER_COLORS = ["#2a78d6", "#eb6834"]  # more capable, less capable
TIER_MARKERS = ["o", "s"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load(results_dir: Path) -> pd.DataFrame:
    rows = []
    for info in list_eval_logs(str(results_dir / "logs")):
        log = read_eval_log(info)
        for s in log.samples or []:
            if not s.scores or "permission_scorer" not in s.scores:
                continue
            scores = s.scores["permission_scorer"].metadata["scores"]
            usage = s.model_usage or {}
            rows.append(
                {
                    "model": log.eval.model,
                    "policy": (log.eval.metadata or {}).get("policy", "model"),
                    "episode": s.id,
                    "template": s.metadata["template"],
                    "r": s.metadata["r"],
                    "consent": s.metadata["consent"],
                    "prompt": s.metadata.get("prompt", "neutral"),
                    "rep": s.metadata["rep"],
                    **scores,
                    "input_tokens": sum(u.input_tokens for u in usage.values()),
                    "output_tokens": sum(u.output_tokens for u in usage.values()),
                    "seconds": s.total_time,
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, prompt, consent, r), g in df.groupby(["model", "prompt", "consent", "r"]):
        n, k = len(g), int(g["unauthorized"].sum())
        lo, hi = wilson(k, n)
        out.append(
            {
                "model": model,
                "prompt": prompt,
                "consent": consent,
                "r": r,
                "n": n,
                "unauthorized": k,
                "uar": k / n,
                "uar_lo": lo,
                "uar_hi": hi,
                "acted_without_asking": g["acted_without_asking"].mean(),
                "asked_owner": g["asked_owner"].mean(),
                "asked_lo": wilson(int(g["asked_owner"].sum()), n)[0],
                "asked_hi": wilson(int(g["asked_owner"].sum()), n)[1],
                "authorized_cancel": g["authorized_cancel"].mean(),
                "legit_completion": g["legit_completion"].mean(),
                "inaction": g["inaction"].mean(),
                "probe_consent_correct": g["probe_consent_correct"].astype(float).mean(),
            }
        )
    return pd.DataFrame(out).sort_values(["model", "prompt", "consent", "r"])


def _label(r: float) -> str:
    return f"{int(r) if float(r).is_integer() else r}×"


SUBTITLES = {
    "neutral": "Owner unreachable before the deadline.",
    "group": "Owner unreachable before the deadline; user asked for as much total value as possible.",
}


METRICS = {
    # column, low, high, y-axis label, title
    "uar": ("uar", "uar_lo", "uar_hi", "Episodes with an unauthorized cancel",
            "How often does an agent cancel another user's job without consent?"),
    "asked": ("asked_owner", "asked_lo", "asked_hi", "Episodes where the agent asked the owner",
              "How often does an agent ask the owner before giving up on the bigger job?"),
}


def plot(
    summary: pd.DataFrame, path: Path, consent: str = "late", prompt: str = "neutral",
    metric: str = "uar",
) -> None:
    col, lo_col, hi_col, ylabel, title = METRICS[metric]
    core = summary[(summary["consent"] == consent) & (summary["prompt"] == prompt)]
    families = {
        fam: [(m, name) for m, name in models if m in set(core["model"])]
        for fam, models in FAMILIES.items()
    }
    families = {fam: ms for fam, ms in families.items() if ms}
    known = {m for ms in families.values() for m, _ in ms}
    extra = [(m, m.split("/")[-1]) for m in sorted(set(core["model"]) - known)]
    if extra:
        families["Other"] = extra[:2]
    if not families:
        return
    levels = sorted(core["r"].unique())
    xs = {r: i for i, r in enumerate(levels)}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(
        1, len(families), figsize=(5.2 * len(families), 4.6), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, (fam, models) in zip(axes[0], families.items()):
        ax.set_facecolor(SURFACE)
        for tier, (model, name) in enumerate(models):
            g = core[core["model"] == model].sort_values("r")
            x = [xs[r] + (tier - 0.5) * 0.08 for r in g["r"]]
            y = g[col] * 100
            color = TIER_COLORS[tier % 2]
            ax.vlines(x, g[lo_col] * 100, g[hi_col] * 100, color=color, lw=1.2, alpha=0.55)
            ax.plot(
                x, y, color=color, lw=2, marker=TIER_MARKERS[tier % 2], ms=8,
                mec=SURFACE, mew=2, label=name, zorder=3,
            )
        ax.set_title(fam, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(range(len(levels)), [_label(r) for r in levels])
        ax.set_xlim(-0.4, len(levels) - 0.6)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        leg = ax.legend(loc="center right", frameon=False, labelcolor=INK)
        for text in leg.get_texts():
            text.set_color(INK)
    axes[0][0].set_ylabel(ylabel, color=INK_2)
    n = int(core["n"].median())
    fig.suptitle(
        title,
        x=0.02, ha="left", color=INK, fontsize=14, fontweight="bold",
    )
    fig.text(
        0.02, 0.905,
        f"{SUBTITLES[prompt]} Preliminary: n = {n} episodes per point, one domain, "
        "bars are 95% intervals.",
        color=INK_2, fontsize=10,
    )
    fig.supxlabel(
        "Value of the blocked job, relative to the job it would displace", color=INK_2, fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mvp")
    df = load(results_dir)
    if df.empty:
        sys.exit(f"no scored episodes found under {results_dir}/logs")
    df.to_csv(results_dir / "episodes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(results_dir / "summary.csv", index=False)
    plot(summary, results_dir / "uar_curve.png")
    plot(summary, results_dir / "asked_owner_curve.png", metric="asked")
    if (summary["prompt"] == "group").any():
        plot(summary, results_dir / "uar_curve_group.png", prompt="group")
    cols = ["model", "prompt", "consent", "r", "n", "uar", "uar_lo", "uar_hi", "asked_owner", "legit_completion"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(
        f"\n{len(df)} episodes, {df['input_tokens'].sum():,} input and "
        f"{df['output_tokens'].sum():,} output tokens, median {df['seconds'].median():.0f}s per episode"
    )


if __name__ == "__main__":
    main()
