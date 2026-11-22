"""Reproduce the Food.com frozen-origin, full-catalog ranking experiment."""

import argparse
import ast
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = {"validation": ("2013-01-01", "2016-01-01"), "test": ("2016-01-01", "2019-01-01")}
HALF_LIVES = ("180", "730", "all")
POPULARITY_WEIGHTS = (0.0, 0.15, 0.35, 0.55, 0.75, 1.0)
CONTENT_WEIGHTS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65)
METHODS = (
    "pop",
    "trend",
    "neighbors",
    "cf",
    "ingredients",
    "raw",
    "hybrid",
    "raw_tuned",
    "no_cf",
    "no_pop",
)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_data(data_dir):
    recipes = pd.read_csv(
        data_dir / "RAW_recipes.csv", usecols=["id", "name", "ingredients", "submitted"]
    )
    events = pd.read_csv(
        data_dir / "RAW_interactions.csv", usecols=["user_id", "recipe_id", "date", "rating"]
    )
    assert recipes.id.is_unique and not recipes.id.isna().any()
    assert not events[["user_id", "recipe_id", "date", "rating"]].isna().any().any()
    assert events.recipe_id.isin(recipes.id).all()
    assert not events.duplicated(["user_id", "recipe_id"]).any()
    stats = dict(
        raw_recipes=len(recipes),
        raw_interactions=len(events),
        raw_users=events.user_id.nunique(),
        recipe_start=recipes.submitted.min(),
        recipe_end=recipes.submitted.max(),
        interaction_start=events.date.min(),
        interaction_end=events.date.max(),
        zero_ratings=int((events.rating == 0).sum()),
    )
    plausible = events.date >= events.recipe_id.map(recipes.set_index("id").submitted)
    stats["reviews_before_submission"] = int((~plausible).sum())
    events = events.loc[plausible].sort_values(["user_id", "date", "recipe_id"])
    stats["clean_ratings"] = len(events)
    stats["clean_positives"] = int((events.rating >= 4).sum())
    stats["clean_users"] = events.user_id.nunique()
    stats["clean_sparsity"] = 1.0 - len(events) / (stats["clean_users"] * len(recipes))
    return recipes, events, stats


def feature_matrices(recipes):
    start = perf_counter()
    idx = {}
    ptr = [0]
    cols = []
    for raw in recipes.ingredients:
        terms = sorted({" ".join(v.lower().split()) for v in ast.literal_eval(raw)} - {""})
        cols.extend(idx.setdefault(term, len(idx)) for term in terms)
        ptr.append(len(cols))
    mat = sparse.csr_matrix(
        (
            np.ones(len(cols), dtype=np.float32),
            np.asarray(cols, np.int32),
            np.asarray(ptr, np.int32),
        ),
        shape=(len(recipes), len(idx)),
    )
    nterms = np.diff(mat.indptr)
    bin_norms = np.sqrt(np.maximum(nterms, 1)).astype(np.float32)
    binary = mat.copy()
    binary.data /= np.repeat(bin_norms, nterms)
    raw_time = perf_counter() - start
    start = perf_counter()
    df = np.bincount(mat.indices, minlength=len(idx))
    idf = (1 + np.log((1 + len(recipes)) / (1 + df))).astype(np.float32)
    mat.data *= idf[mat.indices]
    norm = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    mat.data /= np.repeat(np.maximum(norm, np.finfo(np.float32).tiny), nterms)
    return binary, mat, idx, idf, raw_time, perf_counter() - start


def ranked_top(scores, count=10):
    cut = np.partition(scores, -count)[-count]
    above = np.flatnonzero(scores > cut)
    tied = np.flatnonzero(scores == cut)[: count - len(above)]
    ids = np.r_[above, tied]
    return ids[np.lexsort((ids, -scores[ids]))]


def rank(scores, target):
    v = scores[target]
    return 1 + np.count_nonzero(scores > v) + np.count_nonzero(scores[:target] == v)


def ndcg(rank_value):
    return 1 / np.log2(1 + rank_value) if rank_value <= 10 else 0.0


def profile(matrix, recent):
    vector = np.asarray(matrix[recent].sum(axis=0), dtype=np.float32).ravel()
    vector /= max(np.linalg.norm(vector), np.finfo(np.float32).tiny)
    return matrix @ vector


def collaborative(likes, likes_transpose, u, norm, seen, n_items):
    counts = (likes[u] @ likes_transpose).tocsr()
    ix, c = counts.indices, counts.data
    keep = ix != u
    ix, c = ix[keep], c[keep]
    sim = c / (norm[u] * norm[ix])
    ordering = np.lexsort((ix, -sim))[:80]
    neighbors, strengths = ix[ordering], sim[ordering] ** 2
    selected = likes[neighbors]
    votes = np.bincount(
        selected.indices, weights=np.repeat(strengths, np.diff(selected.indptr)), minlength=n_items
    ).astype(np.float32)
    votes[seen] = 0
    if votes.max() > 0:
        votes /= votes.max()
    return votes, selected


def make_questions(events, catalog, origin, end):
    train = events[events.date < origin].copy()
    positions = pd.Index(catalog.id)
    train["item"] = positions.get_indexer(train.recipe_id)
    assert train.item.min() >= 0
    positives = train[train.rating >= 4]
    history = positives.groupby("user_id", sort=False).item.agg(list)
    seen = train.groupby("user_id", sort=False).item.agg(list)
    ev = events[
        (events.date >= origin)
        & (events.date < end)
        & (events.rating >= 4)
        & events.recipe_id.isin(positions)
        & events.user_id.isin(history.index[history.str.len() >= 5])
    ]
    # One review per user-item pair was verified on the entire raw extract.
    questions = ev.drop_duplicates("user_id", keep="first").copy()
    questions["item"] = positions.get_indexer(questions.recipe_id)
    return train, positives, history, seen, questions


def popularity(positives, origin, n):
    item = positives.item.to_numpy()
    days = (np.datetime64(origin, "D") - positives.date.to_numpy(dtype="datetime64[D]")).astype(int)
    count = np.bincount(item, minlength=n)
    scores = {}
    for h in HALF_LIVES:
        weights = None if h == "all" else np.exp2(-days / float(h))
        total = np.bincount(item, weights=weights, minlength=n)
        transformed = np.log1p(total).astype(np.float32)
        scores[h] = transformed / transformed.max()
    return count, scores


def select_half_life(questions, seen, popularity_scores):
    sums = Counter()
    for q in questions.itertuples():
        s = seen[q.user_id]
        for h, scores in popularity_scores.items():
            r = rank(scores, q.item)
            prior = np.asarray(s)
            r -= np.count_nonzero(
                (scores[prior] > scores[q.item])
                | ((scores[prior] == scores[q.item]) & (prior < q.item))
            )
            sums[h] += ndcg(r)
    means = {h: float(sums[h] / len(questions)) for h in HALF_LIVES}
    return max(HALF_LIVES, key=means.get), means


def run(split, recipes, events, folder, config):
    origin, end = HORIZONS[split]
    started = perf_counter()
    catalog = recipes[recipes.submitted < origin].sort_values("id").reset_index(drop=True)
    original = catalog.id.to_numpy()
    train, positives, histories, seen, questions = make_questions(events, catalog, origin, end)
    n = len(catalog)
    prep = perf_counter() - started
    start = perf_counter()
    freq, pop = popularity(positives, origin, n)
    pop_time = perf_counter() - start
    start = perf_counter()
    users = pd.Index(np.sort(positives.user_id.unique()))
    uidx = users.get_indexer(positives.user_id)
    likes = sparse.csr_matrix(
        (np.ones(len(positives), np.float32), (uidx, positives.item)), shape=(len(users), n)
    )
    likes.sort_indices()
    likes_transpose = likes.T.tocsr()
    norm = np.sqrt(np.diff(likes.indptr) + 10).astype(np.float32)
    cf_time = perf_counter() - start
    binary, weighted, terms, idf, ing_time, idf_time = feature_matrices(catalog)
    example = int(catalog.index[catalog.id == 53235][0])
    info_example = {
        word: {
            "idf": float(idf[terms[word]]),
            "binary_unit": float(binary[example, terms[word]]),
            "weighted_unit": float(weighted[example, terms[word]]),
        }
        for word in ("onion", "wasabi")
    }
    pd.DataFrame({"recipe_id": original, "training_support": freq}).to_csv(
        folder / f"snapshot_{split}.csv", index=False
    )
    cut = int(np.ceil(n * 0.05))
    info = dict(
        origin=origin,
        end_exclusive=end,
        catalog=n,
        train_events=len(train),
        train_users=train.user_id.nunique(),
        train_positives=len(positives),
        train_positive_users=len(users),
        n_questions=len(questions),
        vocabulary=len(terms),
        history_eligible=int((histories.str.len() >= 5).sum()),
        train_sparsity=1.0 - len(train) / (train.user_id.nunique() * n),
        zero_support=int((freq == 0).sum()),
        at_most_4=int((freq <= 4).sum()),
        top5pct_share=float(np.sort(freq)[-cut:].sum() / freq.sum()),
        processed_example_53235=info_example,
        build_s={
            "shared": prep,
            "all_popularities": pop_time,
            "cf": cf_time,
            "ingredient_binary": ing_time,
            "idf_extra": idf_time,
        },
    )
    if split == "validation":
        chosen_half, pop_metrics = select_half_life(questions, seen, pop)
        config = {"half_life": chosen_half, "half_life_validation": pop_metrics}
    trend = pop[config["half_life"]]
    grid = defaultdict(float)
    ledgers, slates = [], []
    timing = Counter()
    for j, q in enumerate(questions.itertuples()):
        u = users.get_loc(q.user_id)
        target = q.item
        prior = np.asarray(seen[q.user_id])
        hist = histories[q.user_id]
        recent = np.array(hist[-20:])
        assert target not in prior and len(hist) >= 5
        t = perf_counter()
        cf, neighbors = collaborative(likes, likes_transpose, u, norm, prior, n)
        timing["cf"] += perf_counter() - t
        t = perf_counter()
        raw = profile(binary, recent)
        timing["raw"] += perf_counter() - t
        t = perf_counter()
        ing = profile(weighted, recent)
        timing["weighted"] += perf_counter() - t
        if split == "validation":
            for p in POPULARITY_WEIGHTS:
                base = (1 - p) * cf + p * trend
                base[prior] = -1.0
                grid[p, 0.0, "weighted"] += ndcg(rank(base, target))
                grid[p, 0.0, "raw"] += ndcg(rank(base, target))
                for a in CONTENT_WEIGHTS[1:]:
                    for encoding, score in (("weighted", ing), ("raw", raw)):
                        mixed = (1 - a) * base + a * score
                        mixed[prior] = -1.0
                        grid[p, a, encoding] += ndcg(rank(mixed, target))
        else:
            p = config["pop_weight"]
            a = config["ingredient_weight"]
            ar = config["raw_tuned_weight"]
            base = (1 - p) * cf + p * trend
            makers = {
                "pop": lambda: pop["all"].copy(),
                "trend": lambda: trend.copy(),
                "neighbors": lambda: cf.copy(),
                "cf": lambda: (1 - p) * cf + p * trend,
                "ingredients": lambda: ing.copy(),
                "raw": lambda: (1 - a) * base + a * raw,
                "hybrid": lambda: (1 - a) * base + a * ing,
                "raw_tuned": lambda: (1 - ar) * base + ar * raw,
                "no_cf": lambda: (1 - a) * p * trend + a * ing,
                "no_pop": lambda: (1 - a) * (1 - p) * cf + a * ing,
            }
            row = {
                "user_id": q.user_id,
                "target_id": q.recipe_id,
                "target_date": q.date,
                "history_length": len(hist),
                "training_support": freq[target],
                "candidate_count": n - len(prior),
                "cf_votes": int(np.count_nonzero(neighbors.indices == target)),
                "neighbor_count": neighbors.shape[0],
                "target_cf_score": float(cf[target]),
                "target_raw_cosine": float(raw[target]),
                "target_weighted_cosine": float(ing[target]),
                "target_pop_score": float(pop["all"][target]),
                "target_trend_score": float(trend[target]),
            }
            for m in METHODS:
                t = perf_counter()
                score = makers[m]()
                score[prior] = -1.0
                top = ranked_top(score)
                timing[f"rank_{m}"] += perf_counter() - t
                row[f"rank_{m}"] = rank(score, target)
                assert (target in top) == (row[f"rank_{m}"] <= 10)
                slates.append([q.user_id, m, *original[top]])
            ledgers.append(row)
        if (j + 1) % 500 == 0:
            print(
                f"{split}: {j + 1}/{len(questions)} ({perf_counter() - started:.0f}s)", flush=True
            )
    if split == "validation":
        best_p = max(POPULARITY_WEIGHTS, key=lambda p: grid[p, 0.0, "weighted"])
        best_including_zero = max(CONTENT_WEIGHTS, key=lambda a: grid[best_p, a, "weighted"])
        best_a = max(CONTENT_WEIGHTS[1:], key=lambda a: grid[best_p, a, "weighted"])
        best_raw = max(CONTENT_WEIGHTS, key=lambda a: grid[best_p, a, "raw"])
        config.update(
            pop_weight=best_p,
            ingredient_weight=best_a,
            raw_tuned_weight=best_raw,
            ingredient_weight_including_zero=best_including_zero,
        )
        dump(folder / "validation_selection.json", config)
        frame = pd.DataFrame(
            [
                dict(pop_weight=p, ingredient_weight=a, encoding=e, ndcg10=v / len(questions))
                for (p, a, e), v in grid.items()
            ]
        )
        frame.to_csv(folder / "validation_grid.csv", index=False)
    else:
        frame = pd.DataFrame(ledgers)
        frame.to_csv(folder / "test_queries.csv", index=False)
        ranked = pd.DataFrame(
            slates, columns=["user_id", "method", *[f"item_{i}" for i in range(1, 11)]]
        )
        ranked.to_csv(folder / "test_top10.csv", index=False)
        summaries = {}
        for m in METHODS:
            r = frame[f"rank_{m}"].to_numpy()
            displayed = ranked[ranked.method == m].iloc[:, 2:].to_numpy().ravel()
            summaries[m] = {
                "hr": float((r <= 10).mean()),
                "ndcg": float(np.where(r <= 10, 1 / np.log2(1 + r), 0).mean()),
                "coverage": len(np.unique(displayed)) / n,
                "unique_recommended": len(np.unique(displayed)),
            }
        dump(folder / "test_summary.json", summaries)
        print(json.dumps(summaries, indent=2), flush=True)
    info["component_ms_per_query"] = {key: 1000 * v / len(questions) for key, v in timing.items()}
    info["wall_s"] = perf_counter() - started
    dump(folder / f"{split}_stats.json", info)
    print(split, config, info, flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--split", choices=["both", "validation", "test"], default="both")
    args = parser.parse_args()
    args.results.mkdir(exist_ok=True, parents=True)
    recipes, events, stats = load_data(args.data_dir)
    stats["environment"] = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        platform=platform.platform(),
    )
    dump(args.results / "data_stats.json", stats)
    cfg = None
    if args.split in ("both", "validation"):
        cfg = run("validation", recipes, events, args.results, None)
    if args.split in ("both", "test"):
        if cfg is None:
            cfg = json.loads((args.results / "validation_selection.json").read_text())
        run("test", recipes, events, args.results, cfg)


if __name__ == "__main__":
    main()
