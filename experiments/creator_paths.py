"""Full-catalogue ranking with a shrinkage estimate of prior preference for a contributor.

Fit the strength of the extra path on the 2013 queries. Use the saved selection verbatim
at both later origins. Graph scores come from the trained semantic_graph scorer.
"""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from run import (
    ROOT,
    feature_matrices,
    load_data,
    make_questions,
    ndcg,
    popularity,
    profile,
    rank,
    ranked_top,
)

DATES = {
    "validation": ("2013-01-01", "2016-01-01"),
    "test": ("2016-01-01", "2019-01-01"),
    "prospective": ("2018-01-01", "2019-01-01"),
}
BONUS_GRID = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
OUT = ROOT / "results/creator_paths"


class CreatorPaths:
    def __init__(self, creator_ids, liked_items):
        if pd.isna(creator_ids).any():
            raise ValueError("Missing contributor identifier")
        self.creator_ids, self.creators = np.unique(creator_ids, return_inverse=True)
        self.creators = self.creators.astype(np.int32)
        prior = np.bincount(self.creators[liked_items], minlength=len(self.creator_ids))
        self.prior = prior.astype(np.float64) / len(liked_items)

    def preferences(self, liked_items):
        counts = np.bincount(self.creators[liked_items], minlength=len(self.creator_ids))
        weights = counts / (counts + 2.0 + len(liked_items) * self.prior)
        return counts, weights.astype(np.float32)


def snapshot(split):
    origin, end = DATES[split]
    recipes, events, _ = load_data(ROOT / "data")
    byline = pd.read_csv(ROOT / "data/RAW_recipes.csv", usecols=["id", "contributor_id"]).set_index(
        "id"
    )
    catalog = recipes.loc[recipes.submitted < origin].sort_values("id").reset_index(drop=True)
    _, positives, history, seen, questions = make_questions(events, catalog, origin, end)
    start = perf_counter()
    paths = CreatorPaths(
        byline.loc[catalog.id, "contributor_id"].to_numpy(), positives.item.to_numpy()
    )
    index_time = perf_counter() - start
    return catalog, positives, history, seen, questions.reset_index(drop=True), paths, index_time


def score_controls(catalog, positives, history, seen, questions, split):
    _, pops = popularity(positives, DATES[split][0], len(catalog))
    binary, weighted, *_ = feature_matrices(catalog)
    for q in questions.itertuples():
        blocked = np.asarray(seen[q.user_id])
        recent = np.asarray(history[q.user_id][-20:])
        raw_profile = profile(binary, recent)
        weighted_profile = profile(weighted, recent)
        scores = {
            "alltime": pops["all"].copy(),
            "two_year": pops["730"].copy(),
            "ingredients_idf": 0.8 * pops["all"] + 0.2 * weighted_profile,
            "ingredients_equal": 0.65 * pops["all"] + 0.35 * raw_profile,
        }
        for values in scores.values():
            values[blocked] = -np.inf
        yield scores


def score_graph(catalog, questions, split):
    from semantic_graph import load_cached_scorer

    scorer = load_cached_scorer(split)
    np.testing.assert_array_equal(scorer.candidate_ids, catalog.id)
    np.testing.assert_array_equal(scorer.user_ids, questions.user_id)
    for offset in range(0, len(questions), 32):
        batch = scorer.score_batch(offset, min(len(questions), offset + 32))
        for scores in batch:
            yield {"adaptive_graph": scores}


def metrics(ranks):
    ranks = np.asarray(ranks)
    hit = ranks <= 10
    return {
        "hits": int(hit.sum()),
        "hr10": float(hit.mean()),
        "ndcg10": float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean()),
        "median_rank": float(np.median(ranks)),
    }


def fit_selection(grid_sums, users):
    validation = {
        name: {str(k): sums[k] / users for k in BONUS_GRID} for name, sums in grid_sums.items()
    }
    # In a tie prefer the smaller change to the already-fitted recommender.
    best = {name: float(max(scores, key=scores.get)) for name, scores in validation.items()}
    return {"grid": validation, "weights": best, "users": users}


def evaluate(split, family):
    catalog, positives, history, seen, questions, paths, index_time = snapshot(split)
    is_validation = split == "validation"
    if family == "controls":
        iterator = score_controls(catalog, positives, history, seen, questions, split)
        augmented = ("alltime", "ingredients_idf")
    else:
        iterator = score_graph(catalog, questions, split)
        augmented = ("adaptive_graph",)
    # The selected validation ranks are cached, avoiding a second semantic scoring pass.
    selected_file = OUT / f"{family}_selection.json"
    selected = {} if is_validation else json.loads(selected_file.read_text())["weights"]
    sums = {name: dict.fromkeys(BONUS_GRID, 0.0) for name in augmented}
    grid_ranks = {name: {k: [] for k in BONUS_GRID} for name in augmented}
    rows, lists = [], []
    recipe_ids = catalog.id.to_numpy()
    prefix = f"{split}_{family}"
    timings = []
    for q, scores in zip(questions.itertuples(), iterator, strict=True):
        target = q.item
        hist = history[q.user_id]
        blocked = np.asarray(seen[q.user_id])
        t0 = perf_counter()
        counts, weights = paths.preferences(hist)
        bonus = weights[paths.creators]
        elapsed = perf_counter() - t0
        author = paths.creators[target]
        row = {
            "user_id": q.user_id,
            "target_id": q.recipe_id,
            "target_date": q.date,
            "candidate_count": len(catalog) - len(blocked),
            "history_length": len(hist),
            "target_creator_id": paths.creator_ids[author],
            "target_creator_prior_likes": counts[author],
            "target_creator_bonus": float(weights[author]),
        }
        for name in augmented:
            base = scores[name]
            if not np.isneginf(base[blocked]).all():
                raise ValueError("Scorer did not mask all previous interactions")
            if not np.isfinite(base[target]):
                raise ValueError("Held-out target must be a valid candidate")
            if is_validation:
                for strength in BONUS_GRID:
                    values = base if strength == 0 else base + strength * bonus
                    r = rank(values, target)
                    grid_ranks[name][strength].append(r)
                    sums[name][strength] += ndcg(r)
            else:
                scores[name + "_creator"] = base + selected[name] * bonus
        if family == "controls":
            bonus[blocked] = -np.inf
            scores["creator_only"] = bonus
        for name, values in scores.items():
            row["rank_" + name] = rank(values, target)
            if not is_validation:
                t0 = perf_counter()
                top = ranked_top(values)
                if family == "graph" and name.endswith("_creator"):
                    elapsed += perf_counter() - t0
                assert len(top) == 10 and (target in top) == (row["rank_" + name] <= 10)
                lists.append([q.user_id, name, *recipe_ids[top]])
        rows.append(row)
        timings.append(1000 * elapsed)
    ledger = pd.DataFrame(rows)
    if is_validation:
        selection = fit_selection(sums, len(ledger))
        for name, strength in selection["weights"].items():
            ledger["rank_" + name + "_creator"] = grid_ranks[name][strength]
        selected_file.write_text(json.dumps(selection, indent=2) + "\n")
    ledger.to_csv(OUT / f"{prefix}_queries.csv", index=False)
    summary = {
        key.removeprefix("rank_"): metrics(ledger[key]) for key in ledger if key.startswith("rank_")
    }
    if lists:
        slates = pd.DataFrame(
            lists, columns=["user_id", "method", *[f"item_{i}" for i in range(1, 11)]]
        )
        slates.to_csv(OUT / f"{prefix}_top10.csv", index=False)
        for method, frame in slates.groupby("method"):
            summary[method]["distinct_recommended"] = len(np.unique(frame.iloc[:, 2:].to_numpy()))
    details = {
        "users": len(ledger),
        "catalogue": len(catalog),
        "date": DATES[split][0],
        "has_prior_creator": int((ledger.target_creator_prior_likes > 0).sum()),
        "creator_index_seconds": index_time,
        "median_creator_operation_ms": float(np.median(timings)),
        "methods": summary,
    }
    (OUT / f"{prefix}_summary.json").write_text(json.dumps(details, indent=2) + "\n")
    print(prefix, json.dumps(details), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["controls", "graph"])
    parser.add_argument("--split", choices=[*DATES, "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    for split_name in DATES if args.split == "all" else [args.split]:
        evaluate(split_name, args.family)
