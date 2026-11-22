"""Rebuild dated evaluation cohorts and verify every graph slate against raw events."""

import argparse
import json

import numpy as np
import pandas as pd
from run import dump, load_data, make_questions, rank, ranked_top
from semantic_graph import (
    BATCH,
    METHODS,
    OUTPUT,
    ROOT,
    SPLITS,
    load_cached_scorer,
    measure,
)


def audit(split, recipes, events):
    origin, end = SPLITS[split]
    catalog = recipes.loc[recipes.submitted < origin].sort_values("id").reset_index(drop=True)
    _, prior, history, seen, query = make_questions(events, catalog, origin, end)
    frame = pd.read_csv(OUTPUT / f"{split}_queries.csv")
    slates = pd.read_csv(OUTPUT / f"{split}_top10.csv")
    scores = load_cached_scorer(split)
    summary = json.loads((OUTPUT / f"{split}_summary.json").read_text())["methods"]
    assert np.array_equal(scores.candidate_ids, catalog.id)
    assert np.array_equal(scores.user_ids, query.user_id)
    assert np.array_equal(frame.user_id, query.user_id)
    assert np.array_equal(frame.target_id, query.recipe_id)
    assert np.array_equal(frame.target_date, query.date)
    assert np.array_equal(frame.history_length, history.loc[query.user_id].str.len())
    n_users, n_items = len(frame), len(catalog)
    assert np.array_equal(frame.candidate_count, n_items - seen.loc[query.user_id].str.len())
    degree = np.bincount(prior.item, minlength=n_items)
    assert np.array_equal(frame.training_support, degree[query.item])
    assert not np.any(scores.id_i[degree == 0])
    seen_keys = np.concatenate(
        [
            index * np.int64(n_items) + np.array(seen[user], dtype=np.int64)
            for index, user in enumerate(query.user_id)
        ]
    )
    for index, user in enumerate(query.user_id):
        begin, stop = scores.seen.indptr[index : index + 2]
        assert np.array_equal(np.sort(seen[user]), np.sort(scores.seen.indices[begin:stop]))
    cols = [f"item_{j}" for j in range(1, 11)]
    assert set(slates.method) == set(METHODS)
    assert len(slates) == len(METHODS) * n_users
    for method in METHODS:
        subset = slates[slates.method == method]
        assert np.array_equal(subset.user_id, scores.user_ids)
        raw = subset[cols].to_numpy()
        indices = catalog.id.searchsorted(raw)
        assert np.array_equal(scores.candidate_ids[indices], raw)
        assert np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0)
        keys = np.arange(n_users, dtype=np.int64)[:, None] * n_items + indices
        assert not np.isin(keys.ravel(), seen_keys).any()
        ranks = frame[f"rank_{method}"].to_numpy()
        assert np.all((ranks >= 1) & (ranks <= frame.candidate_count))
        observed = raw == frame.target_id.to_numpy()[:, None]
        assert np.array_equal(observed.any(axis=1), ranks <= 10)
        hit = ranks <= 10
        assert np.array_equal(observed[hit].argmax(axis=1) + 1, ranks[hit])
        metrics = measure(ranks, raw, n_items)
        assert all(np.isclose(value, summary[method][key]) for key, value in metrics.items())
    frozen = slates.loc[slates.method == "adaptive", cols].to_numpy()
    for begin in range(0, n_users, BATCH):
        stop = min(begin + BATCH, n_users)
        block = scores.score_batch(begin, stop)
        for j, row in enumerate(block, begin):
            positions = ranked_top(row)
            assert np.array_equal(scores.candidate_ids[positions], frozen[j])
            assert rank(row, query.item.iloc[j]) == frame.rank_adaptive.iloc[j]
    return dict(
        split=split,
        queries=n_users,
        catalog=n_items,
        methods=len(METHODS),
        checked_slate_positions=int(10 * len(slates)),
        verified_adaptive_ranks=n_users,
        zero_history_items=int((degree == 0).sum()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    args = parser.parse_args()
    recipes, reviews, _ = load_data(ROOT / "data")
    for name in SPLITS if args.split == "all" else [args.split]:
        report = audit(name, recipes, reviews)
        dump(OUTPUT / f"{name}_audit.json", report)
        print(report, flush=True)
