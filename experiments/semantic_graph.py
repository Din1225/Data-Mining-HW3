"""A dated, text-only extension of TESMR on the complete Food.com catalogue.

The semantic and trainable branches use one LightGCN hop. Learnable ID tables
are optimized using BPR and sampled cross-view InfoNCE. The extension mixes
the two-view score with a popularity prior using historical taste coherence.
"""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from run import (
    HORIZONS,
    ROOT,
    dump,
    load_data,
    make_questions,
    popularity,
    rank,
    ranked_top,
)
from scipy import sparse
from scipy.special import expit, logsumexp

OUTPUT = ROOT / "results/semantic_graph"
EMBEDDINGS = ROOT / "data/cache/recipe_embeddings"
SPLITS = dict(HORIZONS, prospective=("2018-01-01", "2019-01-01"))
SEED = 20261120
DIM = 64
STEPS = 80
BPR_EDGES = 32768
CL_BLOCK = 128
CL_BLOCKS = 16
TEMPERATURE = 0.2
CL_WEIGHT = 0.05
LEARNING_RATE = 0.012
L2 = 1e-4
BATCH = 32
WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
PRIORS = ("all", "730", "180")
METHODS = (
    "title",
    "card",
    "graph_text",
    "learned_id",
    "two_view",
    "fixed",
    "fixed_tuned",
    "adaptive",
    "bpr_graph",
    "adaptive_nocontrast",
    "adaptive_card",
    "adaptive_nolearn",
    "adaptive_notext",
    "adaptive_title",
)
EPS = 1e-12


def unit(matrix):
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(denominator, EPS)


def project_cards(kind, ids):
    all_ids = np.load(EMBEDDINGS / "ids.npy", mmap_mode="r")
    rows = np.searchsorted(all_ids, ids)
    assert np.array_equal(ids, all_ids[rows])
    vectors = np.load(EMBEDDINGS / f"{kind}.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(ids), size=min(24000, len(ids)), replace=False))
    features = vectors[rows[sample]].astype(np.float32)
    center = features.mean(axis=0)
    residual = features - center
    _, eigenvectors = np.linalg.eigh(residual.T @ residual)
    basis = np.ascontiguousarray(eigenvectors[:, -DIM:][:, ::-1])
    pivots = np.argmax(np.abs(basis), axis=0)
    basis *= np.sign(basis[pivots, np.arange(DIM)])
    result = np.empty((len(ids), DIM), dtype=np.float32)
    for start in range(0, len(ids), 8192):
        sl = slice(start, start + 8192)
        result[sl] = unit((vectors[rows[sl]].astype(np.float32) - center) @ basis)
    return result


def relationship_matrix(positives, n):
    users = pd.Index(np.sort(positives.user_id.unique()))
    u = users.get_indexer(positives.user_id).astype(np.int32)
    i = positives.item.to_numpy(dtype=np.int32)
    user_degree = np.bincount(u, minlength=len(users))
    item_degree = np.bincount(i, minlength=n)
    weight = (1.0 / np.sqrt(user_degree[u] * item_degree[i])).astype(np.float32)
    graph = sparse.csr_matrix((weight, (u, i)), shape=(len(users), n))
    assert graph.nnz == len(positives)
    recent = positives.groupby("user_id", sort=False).tail(20)
    recent_u = users.get_indexer(recent.user_id)
    recent_graph = sparse.csr_matrix(
        (np.ones(len(recent), dtype=np.float32), (recent_u, recent.item)), shape=graph.shape
    )
    return users, u, i, graph, recent_graph, item_degree


def recent_profile(recent, cards):
    lengths = np.diff(recent.indptr)
    sums = recent @ cards
    squared_norm = np.einsum("ij,ij->i", sums, sums)
    coherence = (squared_norm - lengths) / np.maximum(lengths * (lengths - 1), 1)
    # Empirical percentiles use all prior warm cooks, not the evaluation cohort.
    references = np.sort(coherence[lengths >= 5])
    quantiles = np.searchsorted(references, coherence, side="right") / len(references)
    return unit(sums), (0.5 + quantiles).astype(np.float32)


def propagate(graph, transpose, user, item):
    return (user + graph @ item) * 0.5, (item + transpose @ user) * 0.5


def sample_negatives(rng, u, items, positive_keys):
    negative = rng.integers(items, size=len(u), dtype=np.int32)
    all_positions = np.arange(len(u))
    while len(all_positions):
        keys = u[all_positions].astype(np.int64) * items + negative[all_positions]
        pos = np.searchsorted(positive_keys, keys)
        existing = positive_keys[np.minimum(pos, len(positive_keys) - 1)] == keys
        all_positions = all_positions[existing]
        negative[all_positions] = rng.integers(items, size=len(all_positions), dtype=np.int32)
    return negative


def contrastive_gradient(gradient, learned, teacher, ids):
    # Distinct positive nodes in each block; the other nodes in it are negatives.
    student = learned[ids].reshape(CL_BLOCKS, CL_BLOCK, DIM)
    norms = np.maximum(np.linalg.norm(student, axis=-1, keepdims=True), EPS)
    student = student / norms
    frozen = teacher[ids].reshape(CL_BLOCKS, CL_BLOCK, DIM)
    logits = (student @ frozen.transpose(0, 2, 1)) / TEMPERATURE
    log_z = logsumexp(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits - log_z)
    diag = np.arange(CL_BLOCK)
    loss = (log_z[:, :, 0] - logits[:, diag, diag]).mean()
    probabilities[:, diag, diag] -= 1
    tangent = (probabilities @ frozen) / (TEMPERATURE * len(ids))
    tangent = (tangent - student * (student * tangent).sum(axis=-1, keepdims=True)) / norms
    gradient[ids] += CL_WEIGHT * tangent.reshape(-1, DIM)
    return float(loss)


def adam_step(table, first, second, gradient, step):
    gradient += (L2 / len(table)) * table
    first *= 0.9
    first += 0.1 * gradient
    second *= 0.999
    second += 0.001 * gradient * gradient
    scale = LEARNING_RATE * np.sqrt(1.0 - 0.999**step) / (1.0 - 0.9**step)
    table -= scale * first / (np.sqrt(second) + 1e-8)


def learn_ids(
    graph, transpose, edge_u, edge_i, sem_u, sem_i, support, use_alignment=True, seed=SEED
):
    n_users, n_items = graph.shape
    rng = np.random.default_rng(seed)
    user = rng.normal(0, 0.03, (n_users, DIM)).astype(np.float32)
    item = rng.normal(0, 0.03, (n_items, DIM)).astype(np.float32)
    cold = support == 0
    item[cold] = 0
    active = np.flatnonzero(~cold)
    user_m, user_v = np.zeros_like(user), np.zeros_like(user)
    item_m, item_v = np.zeros_like(item), np.zeros_like(item)
    target_u, target_i = unit(sem_u), unit(sem_i)
    keys = np.sort(edge_u.astype(np.int64) * n_items + edge_i)
    trace = []
    for step in range(1, STEPS + 1):
        final_u, final_i = propagate(graph, transpose, user, item)
        choice = rng.integers(len(edge_i), size=BPR_EDGES)
        u, pos = edge_u[choice], edge_i[choice]
        neg = sample_negatives(rng, u, n_items, keys)
        difference = final_i[pos] - final_i[neg]
        margin = np.einsum("ij,ij->i", final_u[u], difference)
        coef = (-expit(-margin) / len(u))[:, None]
        grad_u, grad_i = np.zeros_like(user), np.zeros_like(item)
        np.add.at(grad_u, u, coef * difference)
        item_derivative = coef * final_u[u]
        np.add.at(grad_i, pos, item_derivative)
        np.add.at(grad_i, neg, -item_derivative)
        # Identical draws in the BPR-only control keep its triples and initialization matched.
        align_users = rng.choice(n_users, size=CL_BLOCK * CL_BLOCKS, replace=False)
        align_items = rng.choice(active, size=CL_BLOCK * CL_BLOCKS, replace=False)
        clu = cli = 0.0
        if use_alignment:
            clu = contrastive_gradient(grad_u, final_u, target_u, align_users)
            cli = contrastive_gradient(grad_i, final_i, target_i, align_items)
        original_du, original_di = propagate(graph, transpose, grad_u, grad_i)
        adam_step(user, user_m, user_v, original_du, step)
        adam_step(item, item_m, item_v, original_di, step)
        item[cold] = 0
        bpr = float(np.logaddexp(0, -margin).mean())
        trace.append(dict(update=step, bpr=bpr, user_nce=clu, item_nce=cli))
        if step in (1, 20, 40, 60, STEPS):
            print(f"  update {step}/{STEPS} BPR {bpr:.4f}, NCE {clu:.3f}/{cli:.3f}", flush=True)
    return (*propagate(graph, transpose, user, item), trace)


def calibrated(scores, blocked):
    # Scale over exactly the recipes available to each cook.
    out = scores.copy()
    for r, b in enumerate(blocked):
        out[r, b] = np.inf
    min_score = out.min(axis=1, keepdims=True)
    for r, b in enumerate(blocked):
        out[r, b] = -np.inf
    max_score = out.max(axis=1, keepdims=True)
    out -= min_score
    out /= np.maximum(max_score - min_score, EPS)
    for r, b in enumerate(blocked):
        out[r, b] = 0.0
    return out


def gate_weight(base, confidence):
    return base * confidence / (1 - base + base * confidence)


def mask_seen(scores, blocked):
    for r, b in enumerate(blocked):
        scores[r, b] = -np.inf
    return scores


def blend(personal, prior, base, confidence):
    w = np.asarray(gate_weight(base, confidence), dtype=np.float32)
    return (1 - w) * prior + w * personal


class CachedScorer:
    """Aligned raw IDs and deterministic frozen adaptive batch scores, unobserved recipes only."""

    def __init__(self, split):
        path = OUTPUT / "cache" / split
        self.candidate_ids = np.load(path / "recipe_ids.npy", mmap_mode="r")
        self.user_ids = np.load(path / "user_ids.npy", mmap_mode="r")
        self.sem_u = np.load(path / "graph_user.npy", mmap_mode="r")
        self.sem_i = np.load(path / "graph_item.npy", mmap_mode="r")
        self.id_u = np.load(path / "latent_user.npy", mmap_mode="r")
        self.id_i = np.load(path / "latent_item.npy", mmap_mode="r")
        self.gate = np.load(path / "gate.npy", mmap_mode="r")
        self.seen = sparse.load_npz(path / "seen.npz")
        self.configuration = json.loads((OUTPUT / "selection.json").read_text())["adaptive"]
        self.prior = np.load(path / f"prior_{self.configuration['prior']}.npy", mmap_mode="r")

    def score_batch(self, start, stop):
        stop = min(stop, len(self.user_ids))
        if not 0 <= start < stop:
            raise ValueError("scoring range must contain at least one query")
        sl = slice(start, stop)
        blocked = [
            self.seen.indices[self.seen.indptr[j] : self.seen.indptr[j + 1]]
            for j in range(start, stop)
        ]
        scores = self.sem_u[sl] @ self.sem_i.T + self.id_u[sl] @ self.id_i.T
        return mask_seen(
            blend(
                calibrated(scores, blocked),
                self.prior,
                self.configuration["weight"],
                self.gate[sl, None],
            ),
            blocked,
        )


def load_cached_scorer(split):
    return CachedScorer(split)


def save_cache(split, arrays, blocked):
    directory = OUTPUT / "cache" / split
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", values)
    lengths = np.fromiter((len(b) for b in blocked), dtype=np.int32)
    ptr = np.r_[0, np.cumsum(lengths)]
    idx = np.concatenate(blocked)
    seen = sparse.csr_matrix(
        (np.ones(len(idx), np.uint8), idx, ptr), shape=(len(blocked), len(arrays["recipe_ids"]))
    )
    sparse.save_npz(directory / "seen.npz", seen)


def validated_options():
    return [(m, h, w) for m in ("fixed", "adaptive") for h in PRIORS for w in WEIGHTS]


def score_ranks(scores, targets):
    return np.asarray([rank(row, target) for row, target in zip(scores, targets)], dtype=np.int32)


def measure(ranks, titles, n_catalog):
    hits = ranks <= 10
    return dict(
        hits=int(hits.sum()),
        hr=float(hits.mean()),
        ndcg=float(np.where(hits, 1 / np.log2(1 + ranks), 0).mean()),
        unique_titles=int(len(np.unique(titles))),
        coverage=float(len(np.unique(titles)) / n_catalog),
    )


def run_snapshot(split, recipes, events):
    start = perf_counter()
    origin, end = SPLITS[split]
    catalog = recipes.loc[recipes.submitted < origin].sort_values("id").reset_index(drop=True)
    ids = catalog.id.to_numpy()
    _, positives, histories, seen, queries = make_questions(events, catalog, origin, end)
    users, edge_u, edge_i, graph, recent, support = relationship_matrix(positives, len(ids))
    trans = graph.T.tocsr()
    _, priors = popularity(positives, origin, len(ids))
    answers = queries.item.to_numpy()
    query_ids = queries.user_id.to_numpy()
    q = users.get_indexer(query_ids)
    blocked = [np.asarray(seen[x], dtype=np.int32) for x in query_ids]
    timings = {"snapshot_graph_s": perf_counter() - start}

    t = perf_counter()
    title_i = project_cards("title", ids)
    title_u = unit(recent[q] @ title_i)
    card_i = project_cards("card", ids)
    card_u, coherence = recent_profile(recent, card_i)
    timings["projection_and_history_s"] = perf_counter() - t
    t = perf_counter()
    graph_u, graph_i = propagate(graph, trans, card_u, card_i)
    timings["semantic_graph_s"] = perf_counter() - t
    t = perf_counter()
    print(f"{split} training on {len(edge_u):,} positive edges", flush=True)
    latent_u, latent_i, training_trace = learn_ids(
        graph, trans, edge_u, edge_i, graph_u, graph_i, support
    )
    timings["training_s"] = perf_counter() - t
    pd.DataFrame(training_trace).to_csv(OUTPUT / f"{split}_training.csv", index=False)
    t = perf_counter()
    bpr_u, bpr_i, bpr_trace = learn_ids(
        graph, trans, edge_u, edge_i, graph_u, graph_i, support, use_alignment=False
    )
    timings["matched_bpr_training_s"] = perf_counter() - t
    pd.DataFrame(bpr_trace).to_csv(OUTPUT / f"{split}_bpr_training.csv", index=False)
    bpr_u = bpr_u[q].copy()
    gate = coherence[q]
    card_u = card_u[q].copy()
    graph_u = graph_u[q].copy()
    latent_u = latent_u[q].copy()
    save_cache(
        split,
        dict(
            recipe_ids=ids,
            user_ids=query_ids,
            graph_user=graph_u,
            graph_item=graph_i,
            latent_user=latent_u,
            latent_item=latent_i,
            gate=gate,
            **{f"prior_{h}": v for h, v in priors.items()},
        ),
        blocked,
    )
    branches = dict(
        title=(title_u, title_i),
        card=(card_u, card_i),
        graph_text=(graph_u, graph_i),
        learned_id=(latent_u, latent_i),
        bpr_graph=(bpr_u, bpr_i),
    )
    options = validated_options()
    if split == "validation":
        t = perf_counter()
        scores_at = {option: [] for option in options}
        for i in range(0, len(q), BATCH):
            sl = slice(i, i + BATCH)
            two = graph_u[sl] @ graph_i.T + latent_u[sl] @ latent_i.T
            personal = calibrated(two, blocked[sl])
            for method, h, w in options:
                scores = blend(
                    personal, priors[h], w, gate[sl, None] if method == "adaptive" else 1.0
                )
                scores_at[method, h, w].append(
                    score_ranks(mask_seen(scores, blocked[sl]), answers[sl])
                )
        rows = []
        for (method, h, w), chunks in scores_at.items():
            r = np.concatenate(chunks)
            rows.append(
                dict(
                    method=method,
                    prior=h,
                    weight=w,
                    ndcg=np.where(r <= 10, 1 / np.log2(1 + r), 0).mean(),
                    hr=(r <= 10).mean(),
                )
            )
        grid = pd.DataFrame(rows)
        grid.to_csv(OUTPUT / "validation_grid.csv", index=False)
        cfg = {}
        for m in ("fixed", "adaptive"):
            best = (
                grid[grid.method == m].sort_values("ndcg", ascending=False, kind="stable").iloc[0]
            )
            cfg[m] = dict(prior=best.prior, weight=float(best.weight), ndcg=float(best.ndcg))
        dump(OUTPUT / "selection.json", cfg)
        timings["validation_grid_s"] = perf_counter() - t
        print("Selected", cfg, flush=True)
    else:
        cfg = json.loads((OUTPUT / "selection.json").read_text())

    columns = {method: [] for method in METHODS}
    top = {method: [] for method in METHODS}
    component_values = {
        method: []
        for method in (
            "title",
            "card",
            "graph_text",
            "learned_id",
            "two_view",
            "personal",
            "prior",
            "adaptive",
            "gate",
            "personal_weight",
        )
    }
    per_batch = []
    for i in range(0, len(q), BATCH):
        t = perf_counter()
        sl = slice(i, i + BATCH)
        raw = {m: u[sl] @ x.T for m, (u, x) in branches.items()}
        raw["two_view"] = raw["graph_text"] + raw["learned_id"]
        personal = calibrated(raw["two_view"], blocked[sl])
        a = cfg["adaptive"]["weight"]
        prior = priors[cfg["adaptive"]["prior"]]
        confidence = gate[sl, None]
        fixed_cfg = cfg["fixed"]
        raw["fixed"] = blend(personal, prior, a, 1.0)
        raw["fixed_tuned"] = blend(personal, priors[fixed_cfg["prior"]], fixed_cfg["weight"], 1.0)
        raw["adaptive"] = blend(personal, prior, a, confidence)
        raw["adaptive_nocontrast"] = blend(
            calibrated(raw["graph_text"] + raw["bpr_graph"], blocked[sl]), prior, a, confidence
        )
        for ablation, branch in (
            ("adaptive_card", "card"),
            ("adaptive_nolearn", "graph_text"),
            ("adaptive_notext", "learned_id"),
            ("adaptive_title", "title"),
        ):
            raw[ablation] = blend(calibrated(raw[branch], blocked[sl]), prior, a, confidence)
        tgt = answers[sl]
        rr = np.arange(len(tgt))
        for m in ("title", "card", "graph_text", "learned_id", "two_view", "adaptive"):
            component_values[m].extend(raw[m][rr, tgt])
        component_values["personal"].extend(personal[rr, tgt])
        component_values["prior"].extend(prior[tgt])
        component_values["gate"].extend(confidence.ravel())
        component_values["personal_weight"].extend(gate_weight(a, confidence).ravel())
        for method, score in raw.items():
            mask_seen(score, blocked[sl])
            ranks = score_ranks(score, tgt)
            columns[method].extend(ranks)
            batch_top = [ranked_top(row) for row in score]
            assert all(
                (r <= 10) == (target in listed) for r, target, listed in zip(ranks, tgt, batch_top)
            )
            top[method].extend(ids[batch_top])
        per_batch.append(perf_counter() - t)
    frame = pd.DataFrame(
        dict(
            user_id=query_ids,
            target_id=ids[answers],
            target_date=queries.date.to_numpy(),
            candidate_count=[len(ids) - len(s) for s in blocked],
            training_support=support[answers],
            history_length=[len(histories[x]) for x in query_ids],
            **{f"rank_{m}": v for m, v in columns.items()},
            **{f"score_{m}": v for m, v in component_values.items()},
        )
    )
    frame.to_csv(OUTPUT / f"{split}_queries.csv", index=False)
    slates = pd.concat(
        [
            pd.DataFrame(
                dict(
                    user_id=query_ids,
                    method=m,
                    **{f"item_{k + 1}": col for k, col in enumerate(np.asarray(rows).T)},
                )
            )
            for m, rows in top.items()
        ],
        ignore_index=True,
    )
    slates.to_csv(OUTPUT / f"{split}_top10.csv", index=False)
    summary = {m: measure(np.asarray(columns[m]), top[m], len(ids)) for m in METHODS}
    timings["score_all_branches_s"] = sum(per_batch)
    timings["total_s"] = perf_counter() - start
    dump(
        OUTPUT / f"{split}_summary.json",
        dict(
            origin=origin,
            users=len(q),
            catalog=len(ids),
            positive_edges=len(edge_i),
            prior_cooks=len(users),
            training_updates=STEPS,
            contrastive_temperature=TEMPERATURE,
            contrastive_weight=CL_WEIGHT,
            seed=SEED,
            timing=timings,
            methods=summary,
        ),
    )
    print(split, json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS) + ("all",), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    cookbook, reviews, _ = load_data(ROOT / "data")
    for partition in SPLITS if args.split == "all" else [args.split]:
        run_snapshot(partition, cookbook, reviews)
