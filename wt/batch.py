"""Run a stage on several models under a spending rule.

For each model, a few episodes run first. Their token use, priced from OpenRouter's
public price list, gives a projected cost for the full batch. A model whose batch
would cost more than the cap runs on half the scenarios, or is skipped if even that
is too much. The key's remaining limit is checked before each chunk of the main batch.

    python -m wt.batch --stage arena --models openrouter/x-ai/grok-4.7 ... --log-dir results/x/logs
"""

from __future__ import annotations

import argparse
import os

import httpx
from inspect_ai import eval as inspect_eval

from wt.run import build_task
from wt.scenarios import TEMPLATES

API = "https://openrouter.ai/api/v1"


def prices() -> dict[str, tuple[float, float]]:
    data = httpx.get(f"{API}/models", timeout=30).json()["data"]
    return {
        m["id"]: (float(m["pricing"].get("prompt", 0)), float(m["pricing"].get("completion", 0)))
        for m in data
    }


def key_remaining() -> float | None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        data = httpx.get(f"{API}/key", headers={"Authorization": f"Bearer {key}"}, timeout=30).json()
        return data["data"].get("limit_remaining")
    except Exception:
        return None


def cost_of(log, price: tuple[float, float]) -> float:
    total = 0.0
    for s in log.samples or []:
        for u in (s.model_usage or {}).values():
            total += u.input_tokens * price[0] + u.output_tokens * price[1]
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--stage", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--preflight", type=int, default=4, help="episodes to run first, per model")
    p.add_argument("--max-batch-cost", type=float, default=4.0, help="USD cap per model per stage")
    p.add_argument("--stop-below", type=float, default=3.0, help="stop if the key has less than this left")
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--skip-preflight", action="store_true",
                   help="the pre-flight episodes already ran; go straight to the main batch")
    p.add_argument("--half-models", nargs="*", default=[],
                   help="with --skip-preflight: models that run scenarios 1-3 only")
    args = p.parse_args()

    price = prices()
    ids = [s.id for s in build_task(args.stage).dataset]
    step = max(1, len(ids) // args.preflight)
    first = ids[::step][: args.preflight]
    rest = [i for i in ids if i not in first]
    half_keys = {t.key for t in TEMPLATES[:3]}
    rest_half = [i for i in rest if i.split("|")[1 if i.startswith("arena") else 0] in half_keys]

    def run(models: list[str], sample_ids: list[str]):
        return inspect_eval(
            build_task(args.stage),
            model=models,
            sample_id=sample_ids,
            max_samples=args.max_samples,
            max_connections=args.max_samples,
            max_tasks=len(models),
            log_dir=args.log_dir,
            fail_on_error=0.25,
            retry_on_error=2,
            time_limit=900,  # seconds per episode, so one stalled model cannot hold up a batch
            display="none",
            metadata={"stage": args.stage, "policy": "model"},
        )

    print(f"[{args.stage}] key remaining before: {key_remaining()}")
    full, half = [], []
    if args.skip_preflight:
        half = [m for m in args.models if m in args.half_models]
        full = [m for m in args.models if m not in args.half_models]
    for log in [] if args.skip_preflight else run(args.models, first):
        model = log.eval.model
        if log.status != "success":
            print(f"  {model}: preflight failed ({str(log.error.message)[-160:] if log.error else log.status}); skipped")
            continue
        pr = price.get(model.removeprefix("openrouter/"), (0.0, 0.0))
        per_episode = cost_of(log, pr) / max(1, len(log.samples or []))
        projected = per_episode * len(rest)
        if projected <= args.max_batch_cost:
            full.append(model)
            verdict = "full batch"
        elif per_episode * len(rest_half) <= args.max_batch_cost:
            half.append(model)
            verdict = "half batch (3 of 6 scenarios)"
        else:
            verdict = "skipped as too expensive"
        print(f"  {model}: ${per_episode:.3f} per episode, full batch about ${projected:.2f} -> {verdict}")

    # The main batch runs in two chunks (scenarios 1-3, then 4-6) with a budget check before each,
    # so a run can never overshoot the floor by more than one chunk.
    rest_other = [i for i in rest if i not in rest_half]
    for label, models, sample_ids in (
        ("scenarios 1-3", full + half, rest_half),
        ("scenarios 4-6", full, rest_other),
    ):
        if not models or not sample_ids:
            continue
        left = key_remaining()
        if left is not None and left < args.stop_below:
            print(f"key reports {left:.2f} left, below the {args.stop_below:.2f} floor; {label} not run")
            break
        for log in run(models, sample_ids):
            print(f"  {label}: {log.eval.model}: {log.status}, {len(log.samples or [])} episodes")
    print(f"[{args.stage}] key remaining after: {key_remaining()}")


if __name__ == "__main__":
    main()
