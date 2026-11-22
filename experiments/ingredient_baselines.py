"""Score the validation-selected ingredient and popularity baselines at the other forecast dates."""

import json

import numpy as np
import pandas as pd
from run import (
    ROOT,
    feature_matrices,
    load_data,
    make_questions,
    popularity,
    profile,
    rank,
    ranked_top,
)

DEST = ROOT / "results/comparison"
PERIODS = {"validation": ("2013-01-01", "2016-01-01"), "prospective": ("2018-01-01", "2019-01-01")}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    recipes, events, _ = load_data(ROOT / "data")
    cfg = json.loads((ROOT / "results/validation_selection.json").read_text())
    assert cfg["pop_weight"] == 1
    for split, (cut, end) in PERIODS.items():
        catalog = recipes[recipes.submitted.lt(cut)].sort_values("id").reset_index(drop=True)
        ids = catalog.id.to_numpy()
        _, pos, history, seen, queries = make_questions(events, catalog, cut, end)
        freq, priors = popularity(pos, cut, len(ids))
        finite = max((k for k in priors if k != "all"), key=cfg["half_life_validation"].get)
        raw, weighted, *_ = feature_matrices(catalog)
        all_ranks, all_tops = [], []
        for q in queries.itertuples():
            hist = np.asarray(history[q.user_id][-20:])
            prev = np.asarray(seen[q.user_id])
            weight = cfg["ingredient_weight"]
            raw_weight = cfg["raw_tuned_weight"]
            raw_scores = profile(raw, hist)
            weighted_scores = profile(weighted, hist)
            pop = priors[cfg["half_life"]]
            candidates = {
                "pop": priors["all"].copy(),
                "finite_decay": priors[finite].copy(),
                "raw_tuned": (1 - raw_weight) * pop + raw_weight * raw_scores,
                "hybrid": (1 - weight) * pop + weight * weighted_scores,
            }
            result = dict(
                user_id=int(q.user_id),
                target_id=int(q.recipe_id),
                target_date=q.date,
                training_support=int(freq[q.item]),
                history_length=len(history[q.user_id]),
                candidate_count=len(ids) - len(prev),
            )
            for method, scores in candidates.items():
                scores[prev] = -np.inf
                r = int(rank(scores, q.item))
                top = ranked_top(scores)
                assert (r <= 10) == (q.item in top)
                result["rank_" + method] = r
                all_tops.append([q.user_id, method, *ids[top]])
            all_ranks.append(result)
        pd.DataFrame(all_ranks).to_csv(DEST / f"{split}_ingredients_queries.csv", index=False)
        pd.DataFrame(
            all_tops, columns=["user_id", "method", *[f"item_{k}" for k in range(1, 11)]]
        ).to_csv(DEST / f"{split}_ingredients_top10.csv", index=False)
        print(f"{split}: {len(queries):,} queries", flush=True)


if __name__ == "__main__":
    main()
