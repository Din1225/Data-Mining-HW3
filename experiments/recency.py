"""Evaluate the best finite popularity decay separately from the all-time option."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from run import HORIZONS, ROOT, dump, load_data, make_questions, rank


def prepare_pop(positives, origin, size, half_life):
    started = perf_counter()
    weight = None
    if half_life != "all":
        age = np.datetime64(origin, "D") - positives.date.to_numpy(dtype="datetime64[D]")
        weight = np.exp2(-age.astype(int) / float(half_life))
    count = np.bincount(positives.item, weights=weight, minlength=size)
    scores = np.log1p(count).astype(np.float32)
    scores /= scores.max()
    order = np.argsort(-scores, kind="stable")
    return scores, order, perf_counter() - started


def main():
    folder = ROOT / "results"
    val = json.loads((folder / "validation_selection.json").read_text())["half_life_validation"]
    finite = max((h for h in val if h != "all"), key=val.get)
    recipes, events, _ = load_data(ROOT / "data")
    origin, end = HORIZONS["test"]
    catalog = recipes[recipes.submitted < origin].sort_values("id").reset_index(drop=True)
    _, positives, _, seen, questions = make_questions(events, catalog, origin, end)
    popularity, order, build_s = prepare_pop(positives, origin, len(catalog), finite)
    records = []
    slates = []
    original = catalog.id.to_numpy()
    unique = set()
    for q in questions.itertuples():
        prior = np.array(seen[q.user_id])
        v = popularity[q.item]
        ahead_seen = np.count_nonzero(
            (popularity[prior] > v) | ((popularity[prior] == v) & (prior < q.item))
        )
        records.append((q.user_id, q.recipe_id, rank(popularity, q.item) - ahead_seen))
        blocked = set(prior)
        top = []
        for pos in order:
            if pos not in blocked:
                unique.add(pos)
                top.append(original[pos])
                if len(top) == 10:
                    break
        slates.append([q.user_id, "finite_decay", *top])
    pd.DataFrame(
        slates, columns=["user_id", "method", *[f"item_{i}" for i in range(1, 11)]]
    ).to_csv(folder / "finite_decay_top10.csv", index=False)
    rng = np.random.default_rng(2026)
    sample = questions.iloc[rng.choice(len(questions), 256, replace=False)]
    ms = []
    for q in sample.itertuples():
        start = perf_counter()
        blocked = set(seen[q.user_id])
        found = []
        for item in order:
            if item not in blocked:
                found.append(item)
            if len(found) == 10:
                break
        ms.append(1000 * (perf_counter() - start))
    frame = pd.DataFrame(records, columns=["user_id", "target_id", "rank_finite_decay"])
    frame.to_csv(folder / "finite_decay_queries.csv", index=False)
    rank_array = frame.rank_finite_decay.to_numpy()
    scores = np.where(rank_array <= 10, 1 / np.log2(1 + rank_array), 0)
    full = pd.read_csv(folder / "test_queries.csv")
    assert np.array_equal(full.user_id, frame.user_id)
    r = full.rank_hybrid.to_numpy()
    diff = np.where(r <= 10, 1 / np.log2(1 + r), 0) - scores
    rng = np.random.default_rng(20261120)
    boot = np.concatenate(
        [diff[rng.integers(0, len(diff), size=(100, len(diff)))].mean(axis=1) for _ in range(50)]
    )
    dump(
        folder / "finite_decay_summary.json",
        dict(
            half_life_days=int(finite),
            validation_ndcg=val[finite],
            test_ndcg=float(scores.mean()),
            test_hr=float((rank_array <= 10).mean()),
            unique_recommended=len(unique),
            coverage=len(unique) / len(catalog),
            index_build_s=build_s,
            median_ms=float(np.median(ms)),
            hybrid_minus_decay_ndcg=float(diff.mean()),
            paired_bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
        ),
    )


if __name__ == "__main__":
    main()
