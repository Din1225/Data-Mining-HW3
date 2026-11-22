"""Fit a confidence-weighted implicit ALS comparator on each frozen historical catalogue."""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from run import (
    HORIZONS,
    ROOT,
    dump,
    load_data,
    make_questions,
    ndcg,
    popularity,
    rank,
    ranked_top,
)
from scipy import sparse

DEST = ROOT / "results/comparison"
PERIODS = {**HORIZONS, "prospective": ("2018-01-01", "2019-01-01")}
FACTORS = (32, 64)
CONFIDENCE = (5, 30)
PRIOR = ("all", "730")
MIX = (0.10, 0.25, 0.50, 0.75)
SEED = 20261120


def model_data(positive, n_items):
    user_ids, rows = np.unique(positive.user_id.to_numpy(), return_inverse=True)
    cols = positive.item.to_numpy()
    mat = sparse.csr_matrix(
        (np.ones(len(rows), dtype="float32"), (rows, cols)), shape=(len(user_ids), n_items)
    )
    return user_ids, mat


def fit(mat, factors, confidence, threads):
    started = perf_counter()
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=0.1,
        alpha=confidence,
        dtype=np.float32,
        iterations=12,
        num_threads=threads,
        random_state=SEED,
        use_cg=True,
    )
    model.fit(mat, show_progress=False)
    return model, perf_counter() - started


def score(model, ids, questions, seen, recipe_ids, priors, variants, store):
    """Mask every prior rating, count strict dominance and break ties by raw recipe ID."""
    sums = {name: 0.0 for name in variants}
    records, slates = [], []
    ms = []
    users = questions.user_id.to_numpy()
    targets = questions.item.to_numpy()
    for block in range(0, len(users), 48):
        batch = users[block : block + 48]
        batch_scores = model.user_factors[np.searchsorted(ids, batch)] @ model.item_factors.T
        for offset, (user, alsscore) in enumerate(zip(batch, batch_scores)):
            target = targets[block + offset]
            start = perf_counter()
            prior_items = np.asarray(seen[user])
            alsscore[prior_items] = -np.inf
            norm = max(1.0e-6, np.max(alsscore))
            row = {
                "user_id": int(user),
                "target_id": int(recipe_ids[target]),
                "candidate_count": len(recipe_ids) - len(prior_items),
            }
            assert target not in prior_items
            for name, (p, w) in variants.items():
                scores = alsscore if w == 0 else (1 - w) * alsscore / norm + w * priors[p]
                r = int(rank(scores, target))
                sums[name] += ndcg(r)
                if store:
                    top = ranked_top(scores)
                    assert len(set(top).intersection(prior_items)) == 0
                    assert (r <= 10) == (target in top)
                    row[f"rank_{name}"] = r
                    slates.append([int(user), name, *recipe_ids[top]])
            ms.append((perf_counter() - start) * 1000)
            if store:
                records.append(row)
    return {name: v / len(users) for name, v in sums.items()}, records, slates, float(np.median(ms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--split", choices=("all", *PERIODS), default="all")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Use at least one thread.")
    DEST.mkdir(parents=True, exist_ok=True)
    recipes, events, _ = load_data(ROOT / "data")
    for split, (origin, end) in PERIODS.items():
        if args.split not in ("all", split):
            continue
        catalog = recipes[recipes.submitted.lt(origin)].sort_values("id").reset_index(drop=True)
        _, pos, history, seen, questions = make_questions(events, catalog, origin, end)
        ids, mat = model_data(pos, len(catalog))
        _, priors = popularity(pos, origin, len(catalog))
        recipe_ids = catalog.id.to_numpy()
        if split == "validation":
            candidates = []
            for k in FACTORS:
                for c in CONFIDENCE:
                    model, train_s = fit(mat, k, c, args.threads)
                    options = {
                        "als": ("all", 0.0),
                        **{f"{prior}_{w}": (prior, w) for prior in PRIOR for w in MIX},
                    }
                    means, *_ = score(
                        model, ids, questions, seen, recipe_ids, priors, options, False
                    )
                    for name, (prior, w) in options.items():
                        candidates.append(
                            dict(
                                factors=k,
                                confidence=c,
                                prior=prior,
                                prior_share=w,
                                method=name,
                                val_ndcg=means[name],
                                train_s=train_s,
                            )
                        )
                    print(k, c, f"{train_s:.1f}s", means, flush=True)
            grid = pd.DataFrame(candidates)
            grid.to_csv(DEST / "als_validation_grid.csv", index=False)
            selected = {}
            for name, trials in (("als", grid[grid.prior_share.eq(0)]), ("als_blend", grid)):
                best = trials.loc[trials.val_ndcg.idxmax()]
                selected[name] = dict(
                    factors=int(best.factors),
                    confidence=int(best.confidence),
                    prior=best.prior,
                    prior_share=float(best.prior_share),
                    validation_ndcg=float(best.val_ndcg),
                )
            dump(DEST / "als_selection.json", selected)
        selected = json.loads((DEST / "als_selection.json").read_text())
        groups = {}
        for name, params in selected.items():
            key = (params["factors"], params["confidence"])
            groups.setdefault(key, {})[name] = (params["prior"], params["prior_share"])
        merged = None
        slates = []
        timing = {}
        for (k, c), methods in groups.items():
            model, train_s = fit(mat, k, c, args.threads)
            start = perf_counter()
            means, rows, top, _ = score(
                model, ids, questions, seen, recipe_ids, priors, methods, True
            )
            seconds = perf_counter() - start
            timing.update(
                {
                    m: dict(
                        build_s=train_s,
                        batch_rank_ms_per_user=1000 * seconds / len(questions),
                        ndcg=v,
                        shares_batch_cost_with=list(methods),
                    )
                    for m, v in means.items()
                }
            )
            frame = pd.DataFrame(rows)
            keys = ["user_id", "target_id", "candidate_count"]
            merged = (
                frame if merged is None else merged.merge(frame, on=keys, validate="one_to_one")
            )
            slates.extend(top)
        merged.to_csv(DEST / f"{split}_als_queries.csv", index=False)
        pd.DataFrame(
            slates, columns=["user_id", "method", *[f"item_{i}" for i in range(1, 11)]]
        ).to_csv(DEST / f"{split}_als_top10.csv", index=False)
        dump(DEST / f"{split}_als_cost.json", timing)
        print(split, len(questions), timing, flush=True)


if __name__ == "__main__":
    main()
