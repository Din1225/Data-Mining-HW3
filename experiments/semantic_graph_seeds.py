"""Repeat stochastic graph fitting under fixed validation choices; do not retune."""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from run import dump, load_data, make_questions, popularity, rank, ranked_top
from semantic_graph import (
    BATCH,
    OUTPUT,
    ROOT,
    SEED,
    SPLITS,
    blend,
    calibrated,
    learn_ids,
    mask_seen,
    measure,
    project_cards,
    propagate,
    recent_profile,
    relationship_matrix,
)


def repeat(split, recipes, events, config, seeds):
    origin, end = SPLITS[split]
    catalog = recipes.loc[recipes.submitted < origin].sort_values("id").reset_index(drop=True)
    ids = catalog.id.to_numpy()
    _, positives, _, seen, queries = make_questions(events, catalog, origin, end)
    users, eu, ei, graph, recent, support = relationship_matrix(positives, len(ids))
    trans = graph.T.tocsr()
    q = users.get_indexer(queries.user_id)
    block = [np.array(seen[u], dtype=np.int32) for u in queries.user_id]
    _, priors = popularity(positives, origin, len(ids))
    prior = priors[config["prior"]]
    card = project_cards("card", ids)
    taste, coherence = recent_profile(recent, card)
    su, si = propagate(graph, trans, taste, card)
    analyses = []
    ranks = pd.DataFrame(
        dict(user_id=queries.user_id.to_numpy(), target_id=queries.recipe_id.to_numpy())
    )
    for seed in seeds:
        t = perf_counter()
        lu, li, _ = learn_ids(graph, trans, eu, ei, su, si, support, seed=seed)
        observed, slates = [], []
        targets = queries.item.to_numpy()
        for i in range(0, len(q), BATCH):
            sl = slice(i, i + BATCH)
            joined = su[q[sl]] @ si.T + lu[q[sl]] @ li.T
            scores = blend(
                calibrated(joined, block[sl]), prior, config["weight"], coherence[q[sl], None]
            )
            mask_seen(scores, block[sl])
            observed.extend(rank(row, answer) for row, answer in zip(scores, targets[sl]))
            slates.extend(ids[ranked_top(row)] for row in scores)
        ranks[f"rank_seed_{seed}"] = observed
        analyses.append(
            dict(
                seed=seed,
                seconds=perf_counter() - t,
                **measure(np.asarray(observed), slates, len(ids)),
            )
        )
        print(split, analyses[-1], flush=True)
    ranks.to_csv(OUTPUT / f"{split}_extra_seed_ranks.csv", index=False)
    dump(OUTPUT / f"{split}_extra_seeds.json", analyses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "prospective"), required=True)
    args = parser.parse_args()
    data, event, _ = load_data(ROOT / "data")
    cfg = json.loads((OUTPUT / "selection.json").read_text())["adaptive"]
    repeat(args.split, data, event, cfg, (SEED + 1, SEED + 2))
