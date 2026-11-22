"""Validate a score-level mixture of the temporal semantic graph and implicit ALS."""

import argparse
import json

import numpy as np
import pandas as pd
from collaborative_als import DEST as ALS_DIR
from collaborative_als import fit, model_data
from evaluation import flip_test, paired_ci
from run import ROOT, dump, load_data, make_questions, popularity, rank, ranked_top
from semantic_graph import BATCH, SPLITS, load_cached_scorer
from semantic_graph import OUTPUT as GRAPH_DIR

OUTPUT = ROOT / "results/graph_fusion"
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
SLATE = [f"item_{k}" for k in range(1, 11)]


def read_json(path):
    return json.loads(path.read_text())


def graph_rows(graph):
    for i in range(0, len(graph.user_ids), BATCH):
        yield from graph.score_batch(i, i + BATCH)


def als_rows(model, user_ids, questions, seen, prior, weight):
    # Match the frozen comparator's batch width, user-wise peak and operation order.
    for start in range(0, len(questions), 48):
        users = questions[start : start + 48]
        recipe_factors = model.item_factors[: len(prior)]
        scores = model.user_factors[np.searchsorted(user_ids, users)] @ recipe_factors.T
        for user, raw in zip(users, scores):
            raw[seen[user]] = -np.inf
            peak = max(1.0e-6, np.max(raw))
            yield (1 - weight) * raw / peak + weight * prior if weight else raw


def interpolate(graph, als, share):
    if share == 0:
        return graph
    if share == 1:
        return als
    return (1 - share) * graph + share * als


def gains(ranks):
    return np.where(ranks <= 10, 1 / np.log2(1 + ranks), 0)


def run_split(split, recipes, events):
    graph = load_cached_scorer(split)
    origin, stop = SPLITS[split]
    catalog = recipes[recipes.submitted.lt(origin)].sort_values("id")
    recipe_ids = catalog.id.to_numpy()
    _, positives, _, seen, questions = make_questions(events, catalog, origin, stop)
    users = questions.user_id.to_numpy()
    targets = questions.item.to_numpy()
    assert np.array_equal(graph.candidate_ids, recipe_ids)
    assert np.array_equal(graph.user_ids, users)
    selection = read_json(ALS_DIR / "als_selection.json")["als_blend"]
    frozen = OUTPUT / "selection.json"
    if split == "validation":
        shares = SHARES
    else:
        lock = read_json(frozen)
        assert (
            lock["als_configuration"] == selection
            and lock["graph_configuration"] == graph.configuration
        )
        chosen = lock["als_share"]
        shares = sorted({0.0, chosen, 1.0})
    user_ids, history_matrix = model_data(positives, len(catalog))
    model, _ = fit(history_matrix, selection["factors"], selection["confidence"], 2)
    _, priors = popularity(positives, origin, len(catalog))
    als = als_rows(
        model, user_ids, users, seen, priors[selection["prior"]], selection["prior_share"]
    )
    neural = graph_rows(graph)
    ranks = {share: np.empty(len(users), dtype=np.int32) for share in shares}
    slates = {share: np.empty((len(users), 10), dtype=recipe_ids.dtype) for share in shares}
    candidate_counts = []
    for j, (graph_scores, als_scores, uid, target) in enumerate(
        zip(neural, als, users, targets, strict=True)
    ):
        blocked = np.sort(np.asarray(seen[uid]))
        assert np.array_equal(np.flatnonzero(~np.isfinite(graph_scores)), blocked)
        assert np.array_equal(np.flatnonzero(~np.isfinite(als_scores)), blocked)
        assert np.isfinite(graph_scores[target]) and np.isfinite(als_scores[target])
        candidate_counts.append(len(catalog) - len(blocked))
        for share in shares:
            scores = interpolate(graph_scores, als_scores, share)
            r = rank(scores, target)
            items = ranked_top(scores)
            assert np.isfinite(scores[items]).all() and (r <= 10) == (target in items)
            if r <= 10:
                assert target == items[r - 1]
            ranks[share][j] = r
            slates[share][j] = recipe_ids[items]
    summary = []
    for share in shares:
        r = ranks[share]
        n_recipes = len(np.unique(slates[share]))
        summary.append(
            dict(
                als_share=share,
                users=len(users),
                ndcg=float(gains(r).mean()),
                hits=int(np.count_nonzero(r <= 10)),
                hr=float(np.mean(r <= 10)),
                unique_recipes=n_recipes,
                coverage=n_recipes / len(catalog),
                median_rank=float(np.median(r)),
            )
        )
    metrics = pd.DataFrame(summary)
    metrics.to_csv(OUTPUT / f"{split}_metrics.csv", index=False)
    if split == "validation":
        # Shares are sorted ascending; exact validation ties favor the smaller ALS share.
        chosen = float(metrics.loc[metrics.ndcg.idxmax(), "als_share"])
        dump(
            frozen,
            dict(
                als_share=chosen,
                validation_origin=origin,
                metric="NDCG@10",
                share_grid=list(SHARES),
                graph_configuration=graph.configuration,
                als_configuration=selection,
                scoring="weighted sum of native endpoint scores",
                tie_break="lower ALS share",
            ),
        )
    names = dict(adaptive=0.0, als_blend=1.0, fusion=chosen)
    ledger = pd.DataFrame(
        dict(
            user_id=users,
            target_id=recipe_ids[targets],
            candidate_count=candidate_counts,
            **{f"rank_{m}": ranks[w] for m, w in names.items()},
        )
    )
    ledger.to_csv(OUTPUT / f"{split}_queries.csv", index=False)
    top = []
    for method, weight in names.items():
        frame = pd.DataFrame(slates[weight], columns=SLATE)
        frame.insert(0, "method", method)
        frame.insert(0, "user_id", users)
        top.append(frame)
    pd.concat(top).to_csv(OUTPUT / f"{split}_top10.csv", index=False)
    if split != "validation":
        contrasts = []
        delta_groups = pd.factorize(ledger.target_id, sort=True)[0]
        for method, weight in (("adaptive", 0.0), ("als_blend", 1.0)):
            for metric, x, y in (
                ("NDCG@10", gains(ranks[chosen]), gains(ranks[weight])),
                ("HR@10", (ranks[chosen] <= 10).astype(float), (ranks[weight] <= 10).astype(float)),
            ):
                delta = x - y
                lo, hi = paired_ci(delta, np.arange(len(delta)))
                title_lo, title_hi = paired_ci(delta, delta_groups)
                contrasts.append(
                    dict(
                        reference="fusion",
                        comparator=method,
                        metric=metric,
                        delta=float(delta.mean()),
                        user_lo=lo,
                        user_hi=hi,
                        target_lo=title_lo,
                        target_hi=title_hi,
                        sign_flip_p=flip_test(delta),
                    )
                )
        pd.DataFrame(contrasts).to_csv(OUTPUT / f"{split}_contrasts.csv", index=False)
    print(split, "locked ALS share", chosen, metrics.to_string(index=False), flush=True)


def verify(split):
    our_q = pd.read_csv(OUTPUT / f"{split}_queries.csv").set_index("user_id")
    our_t = pd.read_csv(OUTPUT / f"{split}_top10.csv").set_index("user_id")
    for name, base, prefix in (
        ("adaptive", GRAPH_DIR, split),
        ("als_blend", ALS_DIR, f"{split}_als"),
    ):
        reference = pd.read_csv(base / f"{prefix}_queries.csv").set_index("user_id")
        for col in ("target_id", "candidate_count", f"rank_{name}"):
            pd.testing.assert_series_equal(our_q[col], reference[col])
        slates = pd.read_csv(base / f"{prefix}_top10.csv").set_index("user_id")
        pd.testing.assert_frame_equal(
            our_t[our_t.method.eq(name)][SLATE], slates[slates.method.eq(name)][SLATE]
        )
    print(split, "independent ranks, masks and endpoint slates agree", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument(
        "--verify", action="store_true", help="Compare both endpoints to their original ledgers"
    )
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    if args.verify:
        verify(args.split)
    else:
        recipes, events, _ = load_data(ROOT / "data")
        run_split(args.split, recipes, events)
