"""Summarize the live team arena: tables and a chart of how much value each team captured.

    python analysis/arena.py results/arena
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Fixed colour per model, in the palette's categorical order.
MODELS = {
    "claude-haiku-5.5": ("Claude Haiku 5.5", "#2a78d6", "o"),
    "gpt-6.1-sol": ("GPT-6.1 Sol", "#eb6834", "s"),
    "gpt-6.1-terra": ("GPT-6.1 Terra", "#1baf7a", "^"),
    "gemini-4-hydrogen": ("Gemini 4 Hydrogen", "#eda100", "D"),
}


def load(results_dir: Path) -> pd.DataFrame:
    rows = []
    for info in list_eval_logs(str(results_dir / "logs")):
        log = read_eval_log(info)
        for s in log.samples or []:
            if not s.scores or "arena_scorer" not in s.scores:
                continue
            x = dict(s.scores["arena_scorer"].metadata["scores"])
            final = ",".join(x.pop("final_queue"))
            usage = s.model_usage or {}
            rows.append(
                {
                    "model": log.eval.model.split("/")[-1],
                    "episode": s.id,
                    "template": s.metadata["template"],
                    "r": s.metadata["r"],
                    "prompt": s.metadata["prompt"],
                    **x,
                    "final_queue": final,
                    "input_tokens": sum(u.input_tokens for u in usage.values()),
                    "output_tokens": sum(u.output_tokens for u in usage.values()),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "prompt", "r"])
        .agg(
            n=("any_unauthorized", "size"),
            episodes_with_unauthorized_cancel=("any_unauthorized", "sum"),
            fights=("fight", "sum"),
            voluntary_yields=("yields", "sum"),
            high_value_job_ran=("high_job_in", "sum"),
            value_captured=("efficiency", "mean"),
            messages_per_episode=("messages", "mean"),
        )
        .reset_index()
    )


def plot(df: pd.DataFrame, path: Path) -> None:
    levels = sorted(df["r"].unique())
    xs = {r: i for i, r in enumerate(levels)}
    present = [m for m in MODELS if m in set(df["model"])]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for k, model in enumerate(present):
        name, color, marker = MODELS[model]
        g = df[df["model"] == model].groupby("r")["efficiency"].mean()
        x = [xs[r] + (k - (len(present) - 1) / 2) * 0.07 for r in g.index]
        ax.plot(x, g.values * 100, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2,
                label=name, zorder=3)
    ax.set_xticks(range(len(levels)), [f"{int(r)}×" for r in levels])
    ax.set_xlim(-0.4, len(levels) - 0.6)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Value of the one high-value job, relative to the other three", color=INK_2)
    ax.set_ylabel("Share of achievable value the team captured", color=INK_2)
    leg = ax.legend(loc="upper right", frameon=False)
    for text in leg.get_texts():
        text.set_color(INK)
    n = int(df.groupby(["model", "r"]).size().median())
    total = len(df)
    unauth = int(df["any_unauthorized"].sum())
    fig.suptitle("Four agents, two slots: nobody cancels, and the best job stays out",
                 x=0.03, ha="left", color=INK, fontsize=13.5, fontweight="bold")
    fig.text(0.03, 0.895,
             f"{unauth} unauthorized cancels in {total} team episodes. Preliminary: n = {n} per point, "
             "one domain.", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/arena")
    df = load(results_dir)
    if df.empty:
        sys.exit(f"no scored episodes found under {results_dir}/logs")
    df.to_csv(results_dir / "episodes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(results_dir / "summary.csv", index=False)
    plot(df, results_dir / "value_captured.png")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{len(df)} episodes, {int(df['any_unauthorized'].sum())} with an unauthorized cancel")


if __name__ == "__main__":
    main()
