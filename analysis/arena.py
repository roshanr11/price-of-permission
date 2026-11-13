"""Summarize the live team arena: tables and a chart of how much value each team captured.

    python analysis/arena.py results/team
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
# Display names. Colours are assigned per chart, in the palette's fixed order.
NAMES = {
    "claude-haiku-5.5": "Claude Haiku 5.5",
    "claude-fable-5.2": "Claude Fable 5.2",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "gpt-6.1-terra": "GPT-6.1 Terra",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gemini-4-hydrogen": "Gemini 4 Hydrogen",
    "gemini-4-carbon": "Gemini 4 Carbon",
    "muse-wonder": "Muse Wonder",
    "muse-spark-open": "Muse Spark",
    "muse-ember": "Muse Ember",
    "grok-4.7": "Grok 4.7",
    "qwen3.8-max-0902": "Qwen3.8-max",
    "deepseek-v4.1-flash": "DeepSeek v4.1 Flash",
    "kimi-k3.1-pro": "Kimi K3.1 Pro",
    "glm-5.3": "GLM 5.3",
}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def name(model: str) -> str:
    return NAMES.get(model, model)


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
                    "defector": bool(s.metadata.get("defector", False)),
                    "rule": s.metadata.get("rule", "explicit"),
                    **x,
                    "final_queue": final,
                    "input_tokens": sum(u.input_tokens for u in usage.values()),
                    "output_tokens": sum(u.output_tokens for u in usage.values()),
                }
            )
    df = pd.DataFrame(rows)
    # A pre-flight episode can be re-run by a later batch; keep one copy of each.
    df = df.drop_duplicates(subset=["model", "episode"], keep="first").reset_index(drop=True)
    # Logs written before these measures existed have no such fields; they were all zero then.
    for col in ("retaliation", "bystander_enforcement", "copycat", "victim_restored",
                "defector_kept_slot", "over_yield"):
        if col not in df:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model", "rule", "defector", "prompt", "r"])
        .agg(
            n=("any_unauthorized", "size"),
            episodes_with_unauthorized_cancel=("any_unauthorized", "sum"),
            unauthorized_cancels=("unauthorized_count", "sum"),
            fights=("fight", "sum"),
            voluntary_yields=("yields", "sum"),
            over_yields=("over_yield", "sum"),
            high_value_job_ran=("high_job_in", "sum"),
            value_captured=("efficiency", "mean"),
            retaliation=("retaliation", "sum"),
            bystander_enforcement=("bystander_enforcement", "sum"),
            copycat=("copycat", "sum"),
            victim_restored=("victim_restored", "sum"),
            messages_per_episode=("messages", "mean"),
        )
        .reset_index()
    )


def by_model(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(
            episodes=("any_unauthorized", "size"),
            episodes_with_unauthorized_cancel=("any_unauthorized", "sum"),
            unauthorized_cancels=("unauthorized_count", "sum"),
            by_high_value_challenger=("unauthorized_by_high", "sum"),
            by_equal_value_challenger=("unauthorized_by_equal", "sum"),
            by_slot_holder=("unauthorized_by_incumbent", "sum"),
            fights=("fight", "sum"),
            voluntary_yields=("yields", "sum"),
            high_value_job_ran=("high_job_in", "sum"),
            value_captured=("efficiency", "mean"),
        )
        .reset_index()
        .sort_values(["unauthorized_cancels", "value_captured"], ascending=False)
    )


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def line_chart(df, col, path, title, subtitle, ylabel, xlabel, models=None) -> None:
    """Share of episodes (or mean of `col`) at each payoff level, one line per model (at most 8)."""
    models = (models or sorted(df["model"].unique()))[:8]
    levels = sorted(df["r"].unique())
    xs = {r: i for i, r in enumerate(levels)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for k, model in enumerate(models):
        g = df[df["model"] == model].groupby("r")[col].mean()
        x = [xs[r] + (k - (len(models) - 1) / 2) * 0.05 for r in g.index]
        ax.plot(x, g.values * 100, color=PALETTE[k], lw=2, marker=MARKERS[k], ms=8, mec=SURFACE,
                mew=2, label=name(model), zorder=3)
    ax.set_xticks(range(len(levels)), [f"{int(r) if float(r).is_integer() else r}×" for r in levels])
    ax.set_xlim(-0.4, len(levels) - 0.6)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.suptitle(title, x=0.03, ha="left", color=INK, fontsize=13.5, fontweight="bold")
    fig.text(0.03, 0.895, subtitle, color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def bar_chart(table: pd.DataFrame, col: str, path, title, subtitle, xlabel) -> None:
    """One horizontal bar per model: `col` as a share of episodes, with the count written beside it."""
    table = table.sort_values(col)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(table) + 2.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    share = table[col] / table["n"] * 100
    ax.barh([name(m) for m in table["model"]], share, color=PALETTE[0], height=0.62)
    for y, (k, n, v) in enumerate(zip(table[col], table["n"], share)):
        ax.text(v + 1.5, y, f"{int(k)} of {int(n)}", va="center", color=INK_2, fontsize=10)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel(xlabel, color=INK_2)
    for label in ax.get_yticklabels():
        label.set_color(INK)
    fig.suptitle(title, x=0.03, ha="left", color=INK, fontsize=13.5, fontweight="bold")
    fig.text(0.03, 1 - 0.62 / fig.get_figheight(), subtitle, color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/team")
    df = load(results_dir)
    if df.empty:
        sys.exit(f"no scored episodes found under {results_dir}/logs")
    df.to_csv(results_dir / "episodes.csv", index=False)
    summarize(df).to_csv(results_dir / "summary.csv", index=False)
    models = by_model(df)
    models.to_csv(results_dir / "by_model.csv", index=False)

    norule = df[df["rule"] == "none"]
    if not norule.empty:
        t = norule.groupby("model").agg(n=("any_unauthorized", "size"), k=("any_unauthorized", "sum")).reset_index()
        bar_chart(
            t, "k", results_dir / "takeovers_without_rule.png",
            "With the consent rule left out of the policy, who takes another agent's slot?",
            "Share of four-agent episodes with at least one cancel of someone else's job. Preliminary.",
            "Episodes with a takeover",
        )
        swept = norule[norule["prompt"] == "neutral"]
        full_sweep = [m for m, g in swept.groupby("model") if g["r"].nunique() >= 4]
        if full_sweep:
            order = (
                swept[swept["model"].isin(full_sweep)].groupby("model")["any_unauthorized"].mean()
                .sort_values(ascending=False).index.tolist()
            )
            per_point = swept[swept["model"].isin(full_sweep)].groupby(["model", "r"]).size()
            line_chart(
                swept[swept["model"].isin(full_sweep)], "any_unauthorized",
                results_dir / "takeover_curve_without_rule.png",
                "Without a stated consent rule, takeovers rise with the payoff",
                f"Four-agent teams, {per_point.min()}-{per_point.max()} episodes per point (preliminary). "
                "With the rule stated, all three had zero.",
                "Episodes where an agent cancelled another's job",
                "Value of the one high-value job, relative to the other three",
                models=order,
            )
    plain = df[~df["defector"] & (df["prompt"] != "discretion") & (df["rule"] == "explicit")]
    if not plain.empty:
        total, bad = len(plain), int(plain["any_unauthorized"].sum())
        line_chart(
            plain, "efficiency", results_dir / "value_captured.png",
            "Four agents, two slots: how much of the available value did the team capture?",
            f"{bad} of {total} team episodes had an unauthorized cancel. Preliminary, one domain.",
            "Share of achievable value captured",
            "Value of the one high-value job, relative to the other three",
        )
        t = plain.groupby("model").agg(n=("any_unauthorized", "size"), k=("any_unauthorized", "sum")).reset_index()
        bar_chart(
            t, "k", results_dir / "unauthorized_by_model.png",
            "Which models cancel another agent's job without consent?",
            "Share of four-agent team episodes with at least one unauthorized cancel. Preliminary.",
            "Episodes with an unauthorized cancel",
        )
    give = df[df["prompt"] == "discretion"]
    if not give.empty:
        line_chart(
            give, "high_job_in", results_dir / "yield_curve.png",
            "When users allow it, at what payoff do agents give up their slot?",
            "Each user said their own job may wait if another clearly matters more. Preliminary.",
            "Episodes where the high-value job ran",
            "Value of the one high-value job, relative to the other three",
        )
    provoked = df[df["defector"]]
    if not provoked.empty:
        t = provoked.groupby("model").agg(n=("retaliation", "size"), k=("retaliation", "sum")).reset_index()
        bar_chart(
            t, "k", results_dir / "retaliation.png",
            "After a scripted agent takes their slot, do agents take it back?",
            "Share of episodes where the victim cancelled the defector's job. Preliminary.",
            "Episodes with a retaliatory cancel",
        )
    print(models.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{len(df)} episodes, {int(df['any_unauthorized'].sum())} with an unauthorized cancel")


if __name__ == "__main__":
    main()
