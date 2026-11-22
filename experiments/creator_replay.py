"""Measure overlap between forecast origins and time single-cook recommendation."""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from creator_paths import DATES, OUT, metrics, snapshot
from run import ROOT, load_data, ranked_top


def replay():
    recipes, events, _ = load_data(ROOT / "data")
    reports = {}
    queries = {}
    for split, (start, _) in DATES.items():
        queries[split] = pd.read_csv(OUT / f"{split}_controls_queries.csv")
        past = events[events.date < start]
        cards = recipes[recipes.submitted < start]
        all_likes = past[past.rating >= 4]
        observed = queries[split]
        reports[split] = {
            "reviewers": int(past.user_id.nunique()),
            "ratings": len(past),
            "likes": len(all_likes),
            "cards": len(cards),
            "warm_cooks": int((all_likes.user_id.value_counts() >= 5).sum()),
            "queries": len(observed),
            "distinct_targets": int(observed.target_id.nunique()),
            "target_creator_seen": int((observed.target_creator_prior_likes > 0).sum()),
            "median_eligible": float(observed.candidate_count.median()),
            "median_history": float(observed.history_length.median()),
            "median_wait_days": float(
                (pd.to_datetime(observed.target_date) - pd.Timestamp(start)).dt.days.median()
            ),
        }
    old = queries["test"]
    new = queries["prospective"]
    old_pairs = set(zip(old.user_id, old.target_id))
    repeated = [
        tuple(row) in old_pairs
        for row in new[["user_id", "target_id"]].itertuples(index=False, name=None)
    ]
    reports["2016_and_2018"] = {
        "shared_cooks": len(set(old.user_id) & set(new.user_id)),
        "repeated_user_target_pairs": int(sum(repeated)),
        "new_user_target_pairs": int(len(repeated) - sum(repeated)),
    }
    (OUT / "replay.json").write_text(json.dumps(reports, indent=2) + "\n")


def serving():
    from semantic_graph import load_cached_scorer

    catalog, _, histories, _, queries, paths, _ = snapshot("test")
    scorer = load_cached_scorer("test")
    ids = catalog.id.to_numpy()
    np.testing.assert_array_equal(ids, scorer.candidate_ids)
    np.testing.assert_array_equal(queries.user_id, scorer.user_ids)
    weight = json.loads((OUT / "graph_selection.json").read_text())["weights"]["adaptive_graph"]
    stored = pd.read_csv(OUT / "test_graph_top10.csv")
    stored = (
        stored[stored.method == "adaptive_graph_creator"]
        .set_index("user_id")
        .drop(columns="method")
    )
    sample = np.random.default_rng(20261120).choice(len(queries), 205, replace=False)
    elapsed = []
    for n, row in enumerate(sample):
        user = int(queries.iloc[row].user_id)
        start = perf_counter()
        base = scorer.score_batch(row, row + 1)[0]
        _, preferred = paths.preferences(histories[user])
        top_ids = ids[ranked_top(base + weight * preferred[paths.creators])]
        latency = (perf_counter() - start) * 1000
        np.testing.assert_array_equal(top_ids, stored.loc[user])
        if n >= 5:
            elapsed.append(latency)
    metrics = {
        "origin": "2016-01-01",
        "batch_size": 1,
        "warmups": 5,
        "queries": len(elapsed),
        "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "mean_ms": float(np.mean(elapsed)),
        "seed": 20261120,
    }
    (OUT / "serving.json").write_text(json.dumps(metrics, indent=2) + "\n")


def distinct_targets():
    earlier = pd.read_csv(OUT / "test_controls_queries.csv")
    earlier_keys = set(zip(earlier.user_id, earlier.target_id))
    score_files = {
        "author": OUT / "prospective_graph_queries.csv",
        "basic": OUT / "prospective_controls_queries.csv",
        "als": ROOT / "results/comparison/prospective_als_queries.csv",
    }
    fusion = ROOT / "results/graph_fusion/prospective_queries.csv"
    if fusion.exists():
        score_files["fusion"] = fusion
    findings = {}
    for family, path in score_files.items():
        frame = pd.read_csv(path)
        repeated = np.array([key in earlier_keys for key in zip(frame.user_id, frame.target_id)])
        findings[family] = {}
        for column in frame.filter(regex="^rank_"):
            findings[family][column[5:]] = {
                str(label): {"users": int(mask.sum()), **metrics(frame.loc[mask, column])}
                for label, mask in (("new_queries", ~repeated), ("repeated_queries", repeated))
            }
    (OUT / "distinct_queries.json").write_text(json.dumps(findings, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", action="store_true")
    parser.add_argument("--distinct", action="store_true")
    args = parser.parse_args()
    if args.serving:
        serving()
    elif args.distinct:
        distinct_targets()
    else:
        replay()
