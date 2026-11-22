"""Rebuild Tables 1–3 from chronological replay outputs, using only Python's standard library.

CSV rows are tidy: one per origin for Table 1, one per origin and method for
Table 2, and one per case and method for Table 3. Rates and gains are written
at the printed precision (two decimals for HR in percent, five for NDCG).
Differences below that precision do not affect the check unless they change
the rounded value. Counts and case ranks must match exactly.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINS = {
    "validation": ("2013-01-01", "2016-01-01"),
    "test": ("2016-01-01", "2019-01-01"),
    "prospective": ("2018-01-01", "2019-01-01"),
}
# The ledger prefix, internal score name and published label identify each method.
METHODS = [
    ("creator_paths/{split}_controls", "alltime", "All-time popularity"),
    ("creator_paths/{split}_controls", "two_year", "Two-year popularity"),
    ("creator_paths/{split}_controls", "ingredients_equal", "Equal-weight ingredients + popularity"),
    ("creator_paths/{split}_controls", "ingredients_idf", "Rarity-weighted ingredients + popularity"),
    ("comparison/{split}_als", "als", "ALS"),
    ("comparison/{split}_als", "als_blend", "ALS + popularity"),
    ("author_als/{split}", "unit_author_als", "Joint author ALS + popularity (unit strength)"),
    ("semantic_graph/{split}", "adaptive_title", "Title profile + popularity"),
    ("semantic_graph/{split}", "adaptive_card", "Full-card profile + popularity"),
    ("semantic_graph/{split}", "adaptive_nolearn", "Card graph + popularity"),
    ("semantic_graph/{split}", "two_view", "Two-view graph"),
    ("semantic_graph/{split}", "fixed", "Graph + popularity, fixed share"),
    ("semantic_graph/{split}", "adaptive", "Graph + popularity, adaptive share"),
    ("creator_paths/{split}_graph", "adaptive_graph_creator", "Adaptive graph + familiar author"),
]
# Each published example is a 2016 user-target pair, not a new test query.
CASES = [
    ("A", 2194917, 66121, "Mongolian Beef"),
    ("B", 176615, 99476, "Beef and Broccoli Stir Fry"),
    ("C", 572108, 28148, "Chicken Chimichangas"),
]
TABLE_KEYS = {
    "table1_cohorts.csv": ("origin",),
    "table2_rankings.csv": ("origin", "method"),
    "table3_cases.csv": ("origin", "case", "method"),
}


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def cohort_rows(replay):
    rows = []
    for split, (date, _) in ORIGINS.items():
        snapshot = replay[split]
        rows.append({
            "origin": date,
            "published_recipes": snapshot["cards"],
            "prior_likes": snapshot["likes"],
            "cooks_with_five_or_more_likes": snapshot["warm_cooks"],
            "evaluated_cooks": snapshot["queries"],
            "cooks_with_familiar_target_author": snapshot["target_creator_seen"],
        })
    return rows


def method_results(prefix, methods, period, expected_queries, expected_count):
    """Join ranks and recommendation lists and return one rank map and coverage count per method."""
    ranks = {method: {} for method in methods}
    queries = {}
    for row in read_rows(Path(str(prefix) + "_queries.csv")):
        user = int(row["user_id"])
        target = int(row["target_id"])
        candidates = int(row["candidate_count"])
        if user in queries or candidates < 10:
            raise ValueError(f"{prefix}: invalid or duplicate query for user {user}")
        if "target_date" in row and not period[0] <= row["target_date"] < period[1]:
            raise ValueError(f"{prefix}: user {user} has an out-of-window target date {row['target_date']}")
        queries[user] = (target, candidates)
        for method in methods:
            rank = int(row["rank_" + method])
            if not 1 <= rank <= candidates:
                raise ValueError(f"{prefix}: invalid {method} rank {rank} for user {user}")
            ranks[method][user] = rank
    if len(queries) != expected_count:
        raise ValueError(f"{prefix}: expected {expected_count} queries, found {len(queries)}")
    if expected_queries is not None and queries != expected_queries:
        raise ValueError(f"{prefix}: user IDs, target IDs or candidate counts differ across methods")

    exposed = {method: set() for method in methods}
    covered_users = {method: set() for method in methods}
    for row in read_rows(Path(str(prefix) + "_top10.csv")):
        method = row["method"]
        if method not in ranks:
            continue
        user = int(row["user_id"])
        if user not in queries or user in covered_users[method]:
            raise ValueError(f"{prefix}: unexpected or duplicate {method} list for user {user}")
        items = [int(row[f"item_{j}"]) for j in range(1, 11)]
        target, _ = queries[user]
        target_rank = ranks[method][user]
        if len(set(items)) != 10:
            raise ValueError(f"{prefix}: duplicate recommendations for {method}, user {user}")
        if (target_rank <= 10 and items[target_rank - 1] != target) or (target_rank > 10 and target in items):
            raise ValueError(f"{prefix}: {method} target rank contradicts the list for user {user}")
        covered_users[method].add(user)
        exposed[method].update(items)
    for method in methods:
        if covered_users[method] != queries.keys():
            raise ValueError(f"{prefix}: missing {method} lists for {len(queries) - len(covered_users[method])} users")
    return queries, ranks, {method: len(items) for method, items in exposed.items()}


def build_tables(results):
    replay = json.loads((results / "creator_paths/replay.json").read_text(encoding="utf-8"))
    ranking_rows, case_rows = [], []
    families = defaultdict(list)
    for prefix, method, _ in METHODS:
        families[prefix].append(method)
    for split in ("test", "prospective"):
        period = ORIGINS[split]
        queries = None
        all_ranks, all_coverage = {}, {}
        for prefix, methods in families.items():
            filename = results / prefix.format(split=split)
            queries, ranks, coverage = method_results(filename, methods, period, queries, replay[split]["queries"])
            all_ranks.update(ranks)
            all_coverage.update(coverage)
        for _, method, label in METHODS:
            scores = all_ranks[method]
            hits = [rank for rank in scores.values() if rank <= 10]
            gain = math.fsum(1 / math.log2(1 + rank) for rank in hits) / len(scores)
            ranking_rows.append(dict(origin=period[0], method=label, hits=len(hits),
                                     hit_rate_pct=f"{100 * len(hits) / len(scores):.2f}",
                                     ndcg_at_10=f"{gain:.5f}", distinct_recipes=all_coverage[method]))
            if split == "test":
                for case, user, target, name in CASES:
                    if user not in queries or queries[user][0] != target:
                        raise ValueError(f"Case {case}: the expected user-target pair is absent from the 2016 ledger")
                    case_rows.append(dict(origin=period[0], case=case, method=label, user_id=user,
                                          target_id=target, target_name=name, target_rank=scores[user]))
    return {"table1_cohorts.csv": cohort_rows(replay), "table2_rankings.csv": ranking_rows,
            "table3_cases.csv": case_rows}


def index_rows(records, keys):
    indexed = {}
    for row in records:
        key = tuple(row[column] for column in keys)
        if key in indexed:
            raise ValueError(f"Duplicate key {key}")
        indexed[key] = row
    return indexed


def differences(expected_path, actual_path, keys):
    expected = index_rows(read_rows(expected_path), keys)
    actual = index_rows(read_rows(actual_path), keys)
    for key in sorted(expected.keys() | actual.keys()):
        identity = " / ".join(key)
        if key not in expected:
            yield f"{identity}: unexpected row"
        elif key not in actual:
            yield f"{identity}: missing row"
        else:
            ref, got = expected[key], actual[key]
            for column in sorted(ref.keys() | got.keys()):
                if ref.get(column) != got.get(column):
                    yield f"{identity}, {column}: expected {ref.get(column)!r}, got {got.get(column)!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="Root of generated experiment ledgers")
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_tables", help="Directory for the three CSVs")
    parser.add_argument("--check", action="store_true", help="Compare generated rows with the published reference")
    parser.add_argument("--reference", type=Path, default=ROOT / "reference", help="Directory of reference CSVs")
    args = parser.parse_args()
    if args.output.resolve() in {args.reference.resolve(), (ROOT / "reference").resolve()}:
        parser.error("Use an output directory separate from the reference snapshots.")
    try:
        tables = build_tables(args.results)
        args.output.mkdir(parents=True, exist_ok=True)
        errors = 0
        for name, records in tables.items():
            path = args.output / name
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
            print(f"Wrote {path} ({len(records)} rows)")
            if args.check:
                for issue in differences(args.reference / name, path, TABLE_KEYS[name]):
                    errors += 1
                    print(f"{name}: {issue}", file=sys.stderr)
        if args.check and errors == 0:
            print("All three tables match the published values at paper precision.")
        return int(errors > 0)
    except (OSError, KeyError, ValueError) as error:
        print(f"Table generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
