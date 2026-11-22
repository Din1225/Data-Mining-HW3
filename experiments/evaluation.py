"""Reconstruct historical evaluation cohorts and compare full-catalogue rankings."""

import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from run import ROOT, dump

OUT = ROOT / "results/comparison"
RESULTS = ROOT / "results"
PERIODS = {
    "validation": ("2013-01-01", "2016-01-01"),
    "test": ("2016-01-01", "2019-01-01"),
    "prospective": ("2018-01-01", "2019-01-01"),
}
REPLICATES = 5000
SEED = 20261120


def observed_metrics(ranks):
    hit = ranks <= 10
    return hit.astype(float), np.where(hit, 1.0 / np.log2(1 + ranks), 0.0)


def exact_paired_probability(differences):
    """Two-sided exact sign-flip test by splitting nonzero pairs into two halves."""
    values = differences[differences != 0]
    if not len(values):
        return 1.0

    def sign_sums(part):
        sums = np.array([0.0])
        for value in part:
            sums = np.concatenate([sums + value, sums - value])
        return sums

    split = len(values) // 2
    left = sign_sums(values[:split])
    right = np.sort(sign_sums(values[split:]))
    observed = abs(values.sum())
    if observed < 1e-12:
        return 1.0
    # Count both tails; the tolerance retains the observed assignment and its exact ties.
    threshold = observed - max(1e-12, 1e-12 * observed)
    below = np.searchsorted(right, -threshold - left, side="right")
    above = len(right) - np.searchsorted(right, threshold - left, side="left")
    return float((below.sum() + above.sum()) / (len(left) * len(right)))


def reconstruct(typ):
    cut, end = PERIODS[typ]
    cards = pd.read_csv(
        ROOT / "data/RAW_recipes.csv",
        usecols=["id", "submitted", "name", "ingredients", "contributor_id"],
    )
    reviews = pd.read_csv(
        ROOT / "data/RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"]
    )
    assert cards.id.is_unique and not reviews.duplicated(["user_id", "recipe_id"]).any()
    reviews = reviews.merge(
        cards[["id", "submitted"]], left_on="recipe_id", right_on="id", validate="many_to_one"
    )
    reviews = reviews[reviews.date.ge(reviews.submitted) & reviews.submitted.lt(cut)]
    catalog = cards[cards.submitted.lt(cut)].set_index("id")
    past = reviews[reviews.date.lt(cut)]
    positive = past[past.rating.ge(4)].sort_values(["user_id", "date", "recipe_id"])
    lengths = positive.user_id.value_counts()
    future = reviews[
        reviews.user_id.isin(lengths[lengths.ge(5)].index)
        & reviews.date.ge(cut)
        & reviews.date.lt(end)
        & reviews.rating.ge(4)
    ]
    future = future.sort_values(["user_id", "date", "recipe_id"]).drop_duplicates("user_id")
    rows = (
        future[["user_id", "recipe_id", "date"]]
        .rename(columns={"recipe_id": "target_id", "date": "target_date"})
        .copy()
    )
    hist = positive[positive.user_id.isin(rows.user_id)].groupby("user_id").recipe_id.agg(list)
    seen = past[past.user_id.isin(rows.user_id)].groupby("user_id").recipe_id.agg(set)
    counts = positive.recipe_id.value_counts().reindex(catalog.index, fill_value=0)
    rows["training_support"] = rows.target_id.map(counts)
    rows["history_length"] = rows.user_id.map(lengths)
    rows["candidate_count"] = len(catalog) - rows.user_id.map(seen.str.len())
    creator = catalog.contributor_id.to_dict()
    creator_counts = {user: Counter(creator[i] for i in items) for user, items in hist.items()}
    rows["target_creator_prior_likes"] = [
        creator_counts[u][creator[target]] for u, target in zip(rows.user_id, rows.target_id)
    ]
    rows["creator_seen"] = rows.target_creator_prior_likes.gt(0)
    return rows.set_index("user_id"), catalog, seen, hist, counts


def load_files(expected, catalog, seen, paths):
    frame = expected.copy()
    total = {}
    catalog_ids = set(catalog.index)
    for rp, tp in paths:
        ranks = pd.read_csv(rp).set_index("user_id").sort_index()
        assert ranks.index.equals(expected.index), rp
        for c in expected.columns.intersection(ranks.columns):
            assert np.array_equal(expected[c], ranks[c]), (rp, c)
        lists = pd.read_csv(tp).sort_values("user_id")
        for name, subset in lists.groupby("method"):
            rc = "rank_" + name
            values = ranks[rc]
            assert np.array_equal(subset.user_id, ranks.index)
            items = subset[[f"item_{j}" for j in range(1, 11)]].to_numpy()
            if rc in frame:
                assert np.array_equal(frame[rc], values) and np.array_equal(total[name], items), (
                    rp,
                    name,
                )
            for uid, r, gold, slate in zip(subset.user_id, values, ranks.target_id, items):
                assert 1 <= r <= expected.at[uid, "candidate_count"]
                slate_set = set(slate)
                assert (
                    len(slate_set) == 10
                    and slate_set <= catalog_ids
                    and slate_set.isdisjoint(seen[uid])
                ), (rp, uid)
                assert (r <= 10) == (int(gold) in slate_set)
                if r <= 10:
                    assert int(gold) == slate[r - 1], (rp, uid, r)
            frame[rc] = values
            total[name] = items
    for old, repeat in (
        ("pop", "alltime"),
        ("finite_decay", "two_year"),
        ("hybrid", "ingredients_idf"),
        ("raw_tuned", "ingredients_equal"),
        ("adaptive", "adaptive_graph"),
    ):
        if old in total and repeat in total:
            assert np.array_equal(frame["rank_" + old], frame["rank_" + repeat]), (
                old,
                repeat,
                "ranks",
            )
            assert np.array_equal(total[old], total[repeat]), (old, repeat, "slates")
    return frame, total


def paired_ci(delta, groupcodes):
    rng = np.random.default_rng(SEED)
    counts = np.bincount(groupcodes)
    sums = np.bincount(groupcodes, weights=delta)
    n = len(counts)
    draws = np.empty(REPLICATES)
    for offset in range(0, REPLICATES, 64):
        indices = rng.integers(0, n, size=(min(64, REPLICATES - offset), n))
        draws[offset : offset + len(indices)] = sums[indices].sum(axis=1) / counts[indices].sum(
            axis=1
        )
    return np.quantile(draws, [0.025, 0.975]).tolist()


def flip_test(d):
    values = d[d != 0]
    if len(values) <= 34:
        return exact_paired_probability(values)
    rng = np.random.default_rng(SEED)
    simulations = 100000
    extreme = 0
    for j in range(0, simulations, 512):
        signs = (
            rng.integers(0, 2, size=(min(512, simulations - j), len(values)), dtype=np.int8) * 2 - 1
        )
        extreme += np.count_nonzero(abs((signs * values).sum(axis=1)) >= abs(values.sum()) - 1e-12)
    return (extreme + 1) / (simulations + 1)


def summarize(frame, all_lists, support, primary):
    report, contrasts, slices = [], [], []
    extent = len(support)
    group = pd.factorize(frame.target_id, sort=True)[0]
    sizes = frame.training_support
    ref_hr, ref_gain = observed_metrics(frame["rank_" + primary].to_numpy())
    for name, lists in all_lists.items():
        r = frame["rank_" + name].to_numpy()
        hr, gain = observed_metrics(r)
        slots = support.reindex(lists.flatten()).to_numpy()
        report.append(
            dict(
                method=name,
                users=len(r),
                hits=int(hr.sum()),
                hr=float(hr.mean()),
                ndcg=float(gain.mean()),
                coverage=len(np.unique(lists)) / extent,
                unique_recommended=len(np.unique(lists)),
                median_rank=float(np.median(r)),
                displayed_share_support_below20=float(np.mean(slots < 20)),
            )
        )
        for metric, x, ref in (("NDCG@10", gain, ref_gain), ("HR@10", hr, ref_hr)):
            delta = ref - x
            low, hi = paired_ci(delta, np.arange(len(r)))
            title_lo, title_hi = paired_ci(delta, group)
            contrasts.append(
                dict(
                    reference=primary,
                    method=name,
                    metric=metric,
                    delta=float(delta.mean()),
                    user_lo=low,
                    user_hi=hi,
                    target_lo=title_lo,
                    target_hi=title_hi,
                    sign_flip_p=flip_test(delta),
                    improved=int(np.sum(delta > 0)),
                    hurt=int(np.sum(delta < 0)),
                )
            )
        for label, include in {
            "target 0 likes": sizes.eq(0),
            "target 1–19 likes": sizes.between(1, 19),
            "target 20+ likes": sizes.ge(20),
            "creator seen": frame.creator_seen,
            "creator unobserved": ~frame.creator_seen,
            "5–9 past likes": frame.history_length.lt(10),
            "10+ past likes": frame.history_length.ge(10),
        }.items():
            if include.any():
                slices.append(
                    dict(
                        slice=label,
                        method=name,
                        n=int(include.sum()),
                        hits=int(hr[include].sum()),
                        ndcg=float(gain[include].mean()),
                        median_rank=float(np.median(r[include])),
                    )
                )
    contrasts = pd.DataFrame(contrasts)
    for metric in contrasts.metric.unique():
        ix = (
            contrasts[(contrasts.metric == metric) & (contrasts.method != primary)]
            .sort_values("sign_flip_p")
            .index
        )
        p = contrasts.loc[ix, "sign_flip_p"].to_numpy()
        contrasts.loc[ix, "holm_p"] = np.minimum(
            1, np.maximum.accumulate(p * np.arange(len(p), 0, -1))
        )
    return pd.DataFrame(report), contrasts, pd.DataFrame(slices)


def creator_changes(frame, primary, parent):
    """Separate target-specific creator evidence and exposure to other creators' boosts."""
    after = frame["rank_" + primary].to_numpy()
    before = frame["rank_" + parent].to_numpy()
    buckets = frame.target_creator_prior_likes
    out = []
    for group, mask in (
        ("0", buckets.eq(0)),
        ("1", buckets.eq(1)),
        ("2+", buckets.ge(2)),
        ("all", np.ones(len(before), dtype=bool)),
    ):
        after_r, before_r = after[mask], before[mask]
        relative = np.log2(before_r / after_r)
        for cutoff in (1, 10, 100, 1000, 10000):
            gain = (after_r <= cutoff) & (before_r > cutoff)
            loss = (before_r <= cutoff) & (after_r > cutoff)
            out.append(
                dict(
                    creator_prior_likes=group,
                    users=len(after_r),
                    k=cutoff,
                    previous_hits=int(np.count_nonzero(before_r <= cutoff)),
                    extended_hits=int(np.count_nonzero(after_r <= cutoff)),
                    gains=int(gain.sum()),
                    losses=int(loss.sum()),
                    median_log2_rank_ratio=float(np.median(relative)),
                    improved=int(np.count_nonzero(after_r < before_r)),
                    tied=int(np.count_nonzero(after_r == before_r)),
                    hurt=int(np.count_nonzero(after_r > before_r)),
                )
            )
    return pd.DataFrame(out)


def cases(frame, history, catalog, lists, primary, comparator):
    a, b = frame["rank_" + primary], frame["rank_" + comparator]
    categories = {
        "rescued": (a <= 10) & (b > 10),
        "displaced": (a > 10) & (b <= 10),
        "improved_shared_hit": (a < b) & (b <= 10),
        "worsened_shared_hit": (a > b) & (a <= 10),
        "same_creator_gain": (a < b) & frame.creator_seen,
        "unfamiliar_creator_regressions": (a > b) & ~frame.creator_seen,
    }
    out = {}
    for label, mask in categories.items():
        candidate = frame.loc[mask].copy()
        if candidate.empty:
            out[label] = []
            continue
        candidate["distance"] = (np.log2(a) - np.log2(b)).abs()
        candidate = candidate.sort_values(["distance", "target_id"], ascending=[False, True])
        records = []
        for uid, row in candidate.drop_duplicates("target_id").head(6).iterrows():
            item = int(row.target_id)
            target = catalog.loc[item]
            ingredients = set(ast.literal_eval(target.ingredients))
            matches = []
            for rec_id in history[uid]:
                card = catalog.loc[rec_id]
                shared = ingredients & set(ast.literal_eval(card.ingredients))
                matches.append(
                    dict(
                        id=int(rec_id),
                        name=str(card["name"]),
                        count=len(shared),
                        overlap=sorted(shared),
                        same_creator=bool(card.contributor_id == target.contributor_id),
                    )
                )
            pos = frame.index.get_loc(uid)
            old_slate, new_slate = lists[comparator][pos], lists[primary][pos]
            added = []
            for position, rec_id in enumerate(new_slate, 1):
                if rec_id not in old_slate:
                    addition = catalog.loc[rec_id]
                    examples = [
                        m["name"]
                        for m in matches
                        if catalog.at[m["id"], "contributor_id"] == addition.contributor_id
                    ]
                    added.append(
                        dict(
                            rank=position,
                            id=int(rec_id),
                            name=str(addition["name"]),
                            prior_likes_for_creator=len(examples),
                            earlier_recipes=examples[-3:],
                        )
                    )
            records.append(
                dict(
                    user_id=int(uid),
                    date=row.target_date,
                    target_id=item,
                    target_name=str(target["name"]),
                    history=int(row.history_length),
                    support=int(row.training_support),
                    creator_prior_likes=int(row.target_creator_prior_likes),
                    entering_top_ten=added,
                    same_creator_examples=[i for i in matches[::-1] if i["same_creator"]][:3],
                    strongest_recent_ingredient_overlap=max(
                        matches[-20:], key=lambda x: x["count"]
                    ),
                    last_three=matches[-3:],
                    ranks={k[5:]: int(row[k]) for k in frame if k.startswith("rank_")},
                )
            )
        out[label] = records
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=PERIODS, required=True)
    p.add_argument(
        "--primary",
        help="Reference method for paired differences (adaptive with --semantic, otherwise als_blend).",
    )
    p.add_argument(
        "--case-control",
        help="Comparator for query examples (als, or als_blend when als is the reference).",
    )
    p.add_argument(
        "--extra", action="append", nargs=2, default=[], metavar=("QUERIES_CSV", "TOP10_CSV")
    )
    p.add_argument(
        "--semantic", action="store_true", help="Include the frozen semantic_graph ledgers."
    )
    args = p.parse_args()
    if args.primary is None:
        args.primary = "adaptive" if args.semantic else "als_blend"
    if args.case_control is None:
        args.case_control = "als_blend" if args.primary == "als" else "als"
    OUT.mkdir(parents=True, exist_ok=True)
    split = args.split
    if split == "test":
        paths = [
            (RESULTS / "test_queries.csv", RESULTS / "test_top10.csv"),
            (RESULTS / "finite_decay_queries.csv", RESULTS / "finite_decay_top10.csv"),
        ]
    else:
        paths = [(OUT / f"{split}_ingredients_queries.csv", OUT / f"{split}_ingredients_top10.csv")]
    paths.append((OUT / f"{split}_als_queries.csv", OUT / f"{split}_als_top10.csv"))
    if args.semantic:
        paths.append(
            (
                RESULTS / f"semantic_graph/{split}_queries.csv",
                RESULTS / f"semantic_graph/{split}_top10.csv",
            )
        )
    paths.extend((ROOT / a, ROOT / b) for a, b in args.extra)
    expected, catalog, seen, hist, support = reconstruct(split)
    frame, lists = load_files(expected, catalog, seen, paths)
    for flag, name in (("--primary", args.primary), ("--case-control", args.case_control)):
        if name not in lists:
            available = ", ".join(sorted(lists))
            p.error(f"{flag} {name!r} is unavailable for {split}. Available methods: {available}")
    stats, contrasts, slices = summarize(frame, lists, support, args.primary)
    prefix = OUT / f"{split}_{args.primary}"
    stats.to_csv(str(prefix) + "_metrics.csv", index=False)
    contrasts.to_csv(str(prefix) + "_contrasts.csv", index=False)
    slices.to_csv(str(prefix) + "_slices.csv", index=False)
    creator_changes(frame, args.primary, args.case_control).to_csv(
        str(prefix) + "_creator_changes.csv", index=False
    )
    dump(
        Path(str(prefix) + "_cases.json"),
        cases(frame, hist, catalog, lists, args.primary, args.case_control),
    )
    print(
        f"Verified {len(catalog):,} eligible catalogue IDs and {len(frame):,} user-target pairs for {len(lists)} complete full-catalogue methods."
    )
    print(stats[["method", "hits", "ndcg", "unique_recommended"]].to_string(index=False))


if __name__ == "__main__":
    main()
