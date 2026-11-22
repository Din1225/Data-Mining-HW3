"""Reproduce three 2016 query traces from the frozen graph and contributor index."""

import json

import numpy as np
import pandas as pd
from creator_paths import OUT, snapshot
from run import dump, rank, ranked_top
from semantic_graph import BATCH, OUTPUT, load_cached_scorer


def extract():
    catalog, past, history, _, questions, paths, _ = snapshot("test")
    scorer = load_cached_scorer("test")
    assert np.array_equal(catalog.id, scorer.candidate_ids)
    assert np.array_equal(questions.user_id, scorer.user_ids)
    strength = json.loads((OUT / "graph_selection.json").read_text())["weights"]["adaptive_graph"]
    ledger = pd.read_csv(OUT / "test_graph_queries.csv").set_index("user_id")
    components = pd.read_csv(OUTPUT / "test_queries.csv").set_index("user_id")
    slates = pd.read_csv(OUT / "test_graph_top10.csv").set_index(["user_id", "method"])
    cases = []
    for user in (2194917, 176615, 572108):
        q = questions.loc[questions.user_id.eq(user)].iloc[0]
        index = int(q.name)
        offset = index // BATCH * BATCH
        base = scorer.score_batch(offset, offset + BATCH)[index - offset]
        counts, author = paths.preferences(history[user])
        bonus = strength * author[paths.creators]
        final = base + bonus
        item = int(q["item"])
        creator = paths.creators[item]
        source = ledger.loc[user]
        signals = components.loc[user]
        assert int(catalog.id[item]) == int(source.target_id) == int(q.recipe_id)
        assert int(counts[creator]) == int(source.target_creator_prior_likes)
        assert np.isclose(float(author[creator]), source.target_creator_bonus)
        assert np.isclose(float(base[item]), signals.score_adaptive)

        def recipe(j):
            return dict(id=int(catalog.id[j]), title=catalog.name[j])

        def recent(rows, n):
            return [
                dict(**recipe(r.item), date=r.date, stars=int(r.rating))
                for r in rows.tail(n).itertuples()
            ]

        prior = past[past.user_id.eq(user)]
        familiar = prior.loc[paths.creators[prior.item.to_numpy()] == creator]
        case = dict(
            user_id=user,
            eligible_recipes=int(source.candidate_count),
            prior_likes=len(prior),
            latest_likes=recent(prior, 4),
            target=recipe(item),
            target_date=q.date,
            target_rating=int(q.rating),
            target_contributor=int(paths.creator_ids[creator]),
            earlier_likes_for_target_contributor=len(familiar),
            latest_likes_for_target_contributor=recent(familiar, 3),
            personal_weight=float(signals.score_personal_weight),
            graph_personal_score=float(signals.score_personal),
            popularity_score=float(signals.score_prior),
            target_author_score=float(author[creator]),
            validation_coefficient=strength,
            target_weighted_author_bonus=float(bonus[item]),
            target_base_score=float(base[item]),
            target_final_score=float(final[item]),
        )
        for method, name, scores in (
            ("adaptive_graph", "base", base),
            ("adaptive_graph_creator", "final", final),
        ):
            rank_value = int(rank(scores, item))
            assert rank_value == int(source[f"rank_{method}"])
            top = ranked_top(scores)
            assert np.array_equal(catalog.id.to_numpy()[top], slates.loc[(user, method)].to_numpy())
            case[f"{name}_target_rank"] = rank_value
            case[f"{name}_top_ten"] = [
                dict(**recipe(j), rank=k, score=float(scores[j]), author_bonus=float(bonus[j]))
                for k, j in enumerate(top, 1)
            ]
        cases.append(case)
    return dict(
        snapshot="2016-01-01",
        split="test",
        source="frozen full-catalogue scoring",
        selection="three observed examples chosen to inspect individual predictions",
        cases=cases,
    )


if __name__ == "__main__":
    result = extract()
    dump(OUT / "worked_queries.json", result)
    for case in result["cases"]:
        print(
            case["user_id"],
            case["target"]["title"],
            case["target_base_score"],
            case["target_weighted_author_bonus"],
            case["target_final_score"],
            case["base_target_rank"],
            case["final_target_rank"],
        )
