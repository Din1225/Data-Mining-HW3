"""Verify frozen model rows and measure warm ranking from a prebuilt snapshot."""

import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from run import dump, rank, ranked_top
from semantic_graph import (
    OUTPUT,
    SEED,
    SPLITS,
    blend,
    calibrated,
    load_cached_scorer,
    mask_seen,
)


def benchmark(split, sample_size):
    scorer = load_cached_scorer(split)
    queries = pd.read_csv(OUTPUT / f"{split}_queries.csv").set_index("user_id")
    slates = pd.read_csv(OUTPUT / f"{split}_top10.csv").set_index(["user_id", "method"])
    methods = ["graph_text", "learned_id", "two_view", "fixed", "adaptive"]
    times = {method: [] for method in methods}
    seed = np.random.default_rng(SEED)
    sample = seed.choice(
        len(scorer.user_ids), size=min(len(scorer.user_ids), sample_size), replace=False
    )
    a = scorer.configuration["weight"]
    prior = scorer.prior
    for j, index in enumerate(sample):
        user_id = scorer.user_ids[index]
        target_id = queries.at[user_id, "target_id"]
        answer = np.searchsorted(scorer.candidate_ids, target_id)
        blocked = [scorer.seen.indices[scorer.seen.indptr[index] : scorer.seen.indptr[index + 1]]]
        for m in np.roll(methods, j % len(methods)):
            tick = perf_counter()
            sl = slice(index, index + 1)
            if m == "adaptive":
                raw = scorer.score_batch(index, index + 1)
            else:
                if m != "learned_id":
                    raw = scorer.sem_u[sl] @ scorer.sem_i.T
                if m != "graph_text":
                    score = scorer.id_u[sl] @ scorer.id_i.T
                    raw = score if m == "learned_id" else raw + score
                if m == "fixed":
                    raw = blend(calibrated(raw, blocked), prior, a, 1.0)
                raw = mask_seen(raw, blocked)
            indices = ranked_top(raw[0])
            times[m].append(1000 * (perf_counter() - tick))
            expected = slates.loc[(user_id, m)].to_numpy(dtype=np.int64)
            # BLAS can use a different reduction order for one-user and batched products.
            # Identical IDs are expected outside machine-precision ties.
            assert np.array_equal(scorer.candidate_ids[indices], expected), (user_id, m)
            assert abs(rank(raw[0], answer) - queries.at[user_id, f"rank_{m}"]) <= 2
    summary = {
        m: dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)))
        for m, t in times.items()
    }
    return dict(
        split=split,
        users=len(sample),
        seed=SEED,
        methods=summary,
        scope="warm single-user scoring including seen filtering and top-ten selection from prebuilt factors",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=list(SPLITS), default="test")
    parser.add_argument("--users", type=int, default=128)
    args = parser.parse_args()
    report = benchmark(args.split, args.users)
    dump(OUTPUT / f"{args.split}_benchmark.json", report)
    print(report, flush=True)
