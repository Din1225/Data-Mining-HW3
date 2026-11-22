"""Factorize historical cook–recipe and cook–contributor interactions jointly."""

import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from collaborative_als import DEST as ALS_DIR
from collaborative_als import PERIODS, SEED, fit, model_data
from evaluation import flip_test, load_files, paired_ci, reconstruct
from graph_fusion import SLATE, als_rows, gains, read_json
from run import ROOT, dump, load_data, make_questions, popularity, rank, ranked_top
from scipy import sparse

OUT = ROOT / "results/author_als"
STRENGTHS = (0.0, 0.25, 1.0, 4.0)


def author_counts(positive, users, contributors):
    rows = np.searchsorted(users, positive.user_id.to_numpy())
    author_ids, cols = np.unique(contributors[positive.item.to_numpy()], return_inverse=True)
    counts = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(users), len(author_ids))
    )
    assert counts.sum(dtype=np.float64) == len(positive)
    return author_ids, counts


def query_strengths(counts, query_rows, author_ids, target_authors):
    cols = pd.Index(author_ids).get_indexer(target_authors)
    observed = cols >= 0
    prior_likes = np.zeros(len(cols), dtype=np.int32)
    prior_likes[observed] = np.asarray(counts[query_rows[observed], cols[observed]]).ravel()
    return prior_likes


def evaluate(users, targets, seen, recipe_ids, scores):
    ranks = np.empty(len(users), dtype=np.int32)
    slates = np.empty((len(users), 10), dtype=recipe_ids.dtype)
    for j, (uid, target, values) in enumerate(zip(users, targets, scores, strict=True)):
        blocked = np.sort(seen[uid])
        assert np.array_equal(np.flatnonzero(~np.isfinite(values)), blocked) and np.isfinite(
            values[target]
        )
        r = rank(values, target)
        top = ranked_top(values)
        assert np.isfinite(values[top]).all() and (r <= 10) == (target in top)
        if r <= 10:
            assert top[r - 1] == target
        ranks[j] = r
        slates[j] = recipe_ids[top]
    return ranks, slates


def contrast(ledger, method):
    baseline = ledger["rank_als_blend"].to_numpy()
    challenger = ledger["rank_" + method].to_numpy()
    target_groups = pd.factorize(ledger.target_id, sort=True)[0]
    output = []
    for metric, delta in (
        ("NDCG@10", gains(challenger) - gains(baseline)),
        ("HR@10", (challenger <= 10).astype(float) - (baseline <= 10)),
    ):
        lo, hi = paired_ci(delta, np.arange(len(delta)))
        title_lo, title_hi = paired_ci(delta, target_groups)
        output.append(
            dict(
                reference=method,
                comparator="als_blend",
                metric=metric,
                delta=delta.mean(),
                user_lo=lo,
                user_hi=hi,
                target_lo=title_lo,
                target_hi=title_hi,
                sign_flip_p=flip_test(delta),
                improved=int((delta > 0).sum()),
                hurt=int((delta < 0).sum()),
            )
        )
    return output


def audit(split):
    expected, catalog, seen, histories, _ = reconstruct(split)
    ledgers, slates = load_files(
        expected, catalog, seen, [(OUT / f"{split}_queries.csv", OUT / f"{split}_top10.csv")]
    )
    contributors = catalog.contributor_id.to_dict()
    liked = [{contributors[r] for r in histories[u]} for u in ledgers.index]
    n = ledgers.target_creator_prior_likes.to_numpy()
    masks = {
        "all": np.ones(len(n), dtype=bool),
        "unseen_author": n == 0,
        "seen_author": n > 0,
        "one_author_like": n == 1,
        "repeat_author": n >= 2,
    }
    baseline = ledgers.rank_als_blend.to_numpy()
    rows = []
    for name, items in slates.items():
        r = ledgers[f"rank_{name}"].to_numpy()
        familiar = np.array(
            [
                [contributors[recipe] in known for recipe in slate]
                for known, slate in zip(liked, items)
            ]
        )
        for label, mask in masks.items():
            if mask.any():
                rows.append(
                    dict(
                        method=name,
                        target_group=label,
                        users=mask.sum(),
                        hits=(r[mask] <= 10).sum(),
                        ndcg=gains(r[mask]).mean(),
                        median_rank=np.median(r[mask]),
                        slate_familiar_author_share=familiar[mask].mean(),
                        rescued=((r[mask] <= 10) & (baseline[mask] > 10)).sum(),
                        displaced=((r[mask] > 10) & (baseline[mask] <= 10)).sum(),
                    )
                )
    pd.DataFrame(rows).to_csv(OUT / f"{split}_author_exposure.csv", index=False)
    print(
        split, "independent historical reconstruction, masks, ranks and slates verified", flush=True
    )


def run(split, recipes, events):
    origin, end = PERIODS[split]
    params = read_json(ALS_DIR / "als_selection.json")["als_blend"]
    lock = OUT / "selection.json"
    if split == "validation":
        strengths = STRENGTHS
    else:
        selection = read_json(lock)
        assert params == selection["baseline_configuration"] and SEED == selection["seed"]
        chosen = selection["strength"]
        strengths = sorted({0.0, 1.0 if chosen == 0.0 else chosen})
    catalog = recipes[recipes.submitted.lt(origin)].sort_values("id")
    recipe_ids = catalog.id.to_numpy()
    contributors = catalog.contributor_id.to_numpy()
    _, positive, histories, seen, queries = make_questions(events, catalog, origin, end)
    users, interactions = model_data(positive, len(recipe_ids))
    author_ids, counts = author_counts(positive, users, contributors)
    query_ids, targets = queries.user_id.to_numpy(), queries.item.to_numpy()
    query_rows = np.searchsorted(users, query_ids)
    prior_likes = query_strengths(counts, query_rows, author_ids, contributors[targets])
    support, priors = popularity(positive, origin, len(catalog))
    prior = priors[params["prior"]]
    log_counts = counts.copy()
    log_counts.data = np.log2(1 + log_counts.data)
    dump(
        OUT / f"{split}_snapshot.json",
        dict(
            origin=origin,
            end=end,
            prior_users=len(users),
            authors=len(author_ids),
            recipes=len(catalog),
            recipe_edges=interactions.nnz,
            author_edges=counts.nnz,
            repeated_author_edges=int((counts.data >= 2).sum()),
            query_users=len(query_ids),
            previous_author_for_target=int((prior_likes != 0).sum()),
        ),
    )
    all_ranks, all_slates, metrics = {}, {}, []
    for strength in strengths:
        matrix = (
            interactions
            if strength == 0.0
            else sparse.hstack(
                [interactions, strength * log_counts], format="csr", dtype=np.float32
            )
        )
        model, train_s = fit(matrix, params["factors"], params["confidence"], 2)
        t = perf_counter()
        rows = als_rows(model, users, query_ids, seen, prior, params["prior_share"])
        r, top = evaluate(query_ids, targets, seen, recipe_ids, rows)
        all_ranks[strength], all_slates[strength] = r, top
        n_titles = len(np.unique(top))
        metrics.append(
            dict(
                strength=strength,
                users=len(r),
                ndcg=float(gains(r).mean()),
                hits=int((r <= 10).sum()),
                hr=float((r <= 10).mean()),
                median_rank=float(np.median(r)),
                unique_recipes=n_titles,
                coverage=n_titles / len(catalog),
                fit_s=train_s,
                score_s=perf_counter() - t,
            )
        )
        print(split, metrics[-1], flush=True)
        del model, matrix
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / f"{split}_metrics.csv", index=False)
    if split == "validation":
        chosen = float(frame.loc[frame.ndcg.idxmax(), "strength"])
        dump(
            lock,
            dict(
                strength=chosen,
                strength_grid=list(STRENGTHS),
                validation_origin=origin,
                metric="NDCG@10",
                tie_break="lower contributor strength",
                seed=SEED,
                baseline_configuration=params,
                author_confidence="strength * log2(1 + prior likes for the contributor), before global alpha",
                diagnostic_if_zero=1.0,
            ),
        )
    names = dict(als_blend=0.0, author_als=chosen)
    if chosen == 0.0:
        names["unit_author_als"] = 1.0
    ledger = pd.DataFrame(
        dict(
            user_id=query_ids,
            target_id=recipe_ids[targets],
            target_date=queries.date.to_numpy(),
            candidate_count=[len(catalog) - len(seen[u]) for u in query_ids],
            training_support=support[targets],
            history_length=[len(histories[u]) for u in query_ids],
            target_creator_prior_likes=prior_likes,
            **{f"rank_{m}": all_ranks[s] for m, s in names.items()},
        )
    )
    ledger.to_csv(OUT / f"{split}_queries.csv", index=False)
    all_lists = []
    for name, strength in names.items():
        lists = pd.DataFrame(all_slates[strength], columns=SLATE)
        lists.insert(0, "method", name)
        lists.insert(0, "user_id", query_ids)
        all_lists.append(lists)
    slates = pd.concat(all_lists)
    slates.to_csv(OUT / f"{split}_top10.csv", index=False)
    zero = pd.read_csv(ALS_DIR / f"{split}_als_queries.csv")
    for col in ("user_id", "target_id", "candidate_count", "rank_als_blend"):
        np.testing.assert_array_equal(zero[col], ledger[col])
    zero = (
        pd.read_csv(ALS_DIR / f"{split}_als_top10.csv")
        .query("method == 'als_blend'")
        .set_index("user_id")
    )
    pd.testing.assert_frame_equal(
        zero[SLATE], slates.query("method == 'als_blend'").set_index("user_id")[SLATE]
    )
    if split != "validation":
        paired = [row for name in names if name != "als_blend" for row in contrast(ledger, name)]
        pd.DataFrame(paired).to_csv(OUT / f"{split}_contrasts.csv", index=False)
    print(split, "selected strength", chosen, "baseline ranks and slates verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(PERIODS), required=True)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Reconstruct eligibilities and examine contributor exposure",
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.audit:
        audit(args.split)
    else:
        recipes, events, _ = load_data(ROOT / "data")
        authors = pd.read_csv(ROOT / "data/RAW_recipes.csv", usecols=["id", "contributor_id"])
        assert not authors.contributor_id.isna().any()
        recipes = recipes.merge(authors, on="id", validate="one_to_one")
        run(args.split, recipes, events)
