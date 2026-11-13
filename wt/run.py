"""Run the experiment matrix.

Examples:
    python -m wt.run --stage smoke --models openrouter/anthropic/claude-haiku-5.5
    python -m wt.run --stage mvp
    python -m wt.run --stage baseline --policy never_act      # no API calls
    python -m wt.run --stage arena                            # live four-agent teams
"""

from __future__ import annotations

import argparse

from inspect_ai import eval as inspect_eval

from wt.arena import team_arena
from wt.task import price_of_permission

# Model ids as listed by OpenRouter (GET https://openrouter.ai/api/v1/models).
MVP_MODELS = [
    "openrouter/anthropic/claude-haiku-5.5",
    "openrouter/anthropic/claude-fable-5.2",
    "openrouter/google/gemini-4-hydrogen",
    "openrouter/google/gemini-4-carbon",
]


def build_task(stage: str, policy: str | None = None, max_turns: int = 12):
    if stage.startswith("arena"):
        return team_arena(stage=stage, policy=policy)
    return price_of_permission(stage=stage, policy=policy, max_turns=max_turns)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", default="mvp", choices=["mvp", "mvp_group", "hard", "max", "smoke", "smoke_hard", "baseline", "arena", "arena_defector", "arena_give", "arena_norule", "arena_norule_sweep", "arena_norule_more", "arena_smoke"])
    p.add_argument("--models", nargs="+", default=MVP_MODELS)
    p.add_argument("--policy", choices=["never_act", "always_maximize", "ask_first", "polite", "grab", "accept", "retaliate", "yield"])
    p.add_argument("--limit", type=int, help="run only the first N episodes per model")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-samples", type=int, default=16, help="episodes in parallel per model")
    p.add_argument("--max-connections", type=int, default=16)
    p.add_argument("--log-dir", default=None)
    args = p.parse_args()

    log_dir = args.log_dir or f"results/{args.stage}/logs"
    models = ["mockllm/model"] if args.policy else args.models
    the_task = build_task(args.stage, args.policy, args.max_turns)
    logs = inspect_eval(
        the_task,
        model=models,
        limit=args.limit,
        max_samples=args.max_samples,
        max_connections=args.max_connections,
        max_tasks=len(models),
        log_dir=log_dir,
        fail_on_error=0.2,
        retry_on_error=2,
        metadata={"stage": args.stage, "policy": args.policy or "model"},
    )
    for log in logs:
        print(f"{log.eval.model}: {log.status}, {len(log.samples or [])} episodes -> {log.location}")


if __name__ == "__main__":
    main()
