"""
用途：
    第 2 部分（主實驗、ensemble、超參數、消融實驗）的交付表格與預測檔。
    接續隊友 v1 的切點（2013 驗證 / 2016 測試 / 2018 回測），只讀 results/ 已跑好的逐使用者結果，
    不重新訓練任何模型；顯著性檢定與錯誤案例分析留給第 3 部分。

主要流程：
    1. 讀各方法的逐使用者排名（*_queries.csv）與前十名清單（*_top10.csv），核對每個檔案的題目一致
    2. 主實驗表：16 個方法（含 1 個 ensemble）在 2016 測試與 2018 回測的 HR@10、NDCG@10、涵蓋食譜數、中位名次
    3. 消融表：A 主方法一次拿掉一個零件；B 文字特徵階梯（菜名 → 整張卡 → 互動圖）；
       C 從整張卡拿掉食材或標籤（需先跑 experiments/card_field_ablation.py，沒跑就留空）
    4. 超參數表：每個方法在 2013 驗證集試過哪些值、選了哪個（另附完整搜尋紀錄）
    5. 隨機種子表、成本表
    6. 匯出逐使用者名次與前十名清單（給第 3 部分），並把所有表格整理成一份 Markdown

執行（repo 根目錄，先跑完 scripts/reproduce.py 產生 results/）：
    FOODCOM_DATA_DIR=data conda run -n dm-hw3 python scripts/main_experiment_tables.py
輸出：main_experiments/
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import semantic_graph as sg  # noqa: E402  只讀取學習 ID 模型的固定超參數常數

RESULTS = ROOT / "results"
OUT = ROOT / "main_experiments"
DATA = Path(os.environ["FOODCOM_DATA_DIR"])
SPLITS = {"test": "2016-01-01", "prospective": "2018-01-01"}
PRIMARY = "adaptive_graph_creator"
ITEMS = [f"item_{j}" for j in range(1, 11)]

# 方法代號 → (論文名稱, 類別, 結果檔前綴)。名次在 {前綴}_queries.csv 的 rank_<代號> 欄，前十名在 {前綴}_top10.csv
METHODS = {
    "alltime": ("All-time popularity", "Non-personalised", "creator_paths/{split}_controls"),
    "two_year": ("Two-year popularity (730-day half-life)", "Non-personalised", "creator_paths/{split}_controls"),
    "neighbors": ("User-kNN (80 neighbours)", "Collaborative filtering", "{split}"),
    "als": ("ALS", "Collaborative filtering", "comparison/{split}_als"),
    "als_blend": ("ALS + popularity", "Collaborative filtering", "comparison/{split}_als"),
    "unit_author_als": ("Joint author ALS + popularity (unit strength)", "Collaborative filtering", "author_als/{split}"),
    "ingredients_equal": ("Equal-weight ingredients + popularity", "Content + popularity", "creator_paths/{split}_controls"),
    "ingredients_idf": ("Rarity-weighted ingredients + popularity", "Content + popularity", "creator_paths/{split}_controls"),
    "adaptive_title": ("Title profile + popularity", "Content + popularity", "semantic_graph/{split}"),
    "adaptive_card": ("Full-card profile + popularity", "Content + popularity", "semantic_graph/{split}"),
    "adaptive_nolearn": ("Card graph + popularity", "Semantic graph", "semantic_graph/{split}"),
    "two_view": ("Two-view graph", "Semantic graph", "semantic_graph/{split}"),
    "fixed": ("Graph + popularity, fixed share", "Semantic graph", "semantic_graph/{split}"),
    "adaptive": ("Graph + popularity, adaptive share", "Semantic graph", "semantic_graph/{split}"),
    "fusion": ("Graph + ALS fusion", "Ensemble", "graph_fusion/{split}"),
    "adaptive_graph_creator": ("Adaptive graph + familiar author (main)", "Main method", "creator_paths/{split}_graph"),
    # 以下只出現在消融表
    "adaptive_notext": ("Learned IDs + popularity", "Ablation only", "semantic_graph/{split}"),
    "adaptive_nocontrast": ("Graph + BPR-only IDs + popularity", "Ablation only", "semantic_graph/{split}"),
}
MAIN_TABLE = list(METHODS)[:16]
# 食譜卡欄位消融（experiments/card_field_ablation.py 的結果）：代號 → (論文名稱, 卡片版本, 結果檔裡的原方法代號)
CARD_FIELDS = {
    "card_profile_no_ingredients": ("Full-card profile − ingredients", "card_no_ingredients", "adaptive_card"),
    "card_profile_no_tags": ("Full-card profile − tags", "card_no_tags", "adaptive_card"),
    "card_graph_no_ingredients": ("Card graph − ingredients", "card_no_ingredients", "adaptive_nolearn"),
    "card_graph_no_tags": ("Card graph − tags", "card_no_tags", "adaptive_nolearn"),
    "graph_model_no_ingredients": ("Graph model − ingredients", "card_no_ingredients", "adaptive"),
    "graph_model_no_tags": ("Graph model − tags", "card_no_tags", "adaptive"),
}
for _key, (_label, _variant, _) in CARD_FIELDS.items():
    METHODS[_key] = (_label, "Ablation only", f"card_fields/{_variant}/{{split}}")


# 結果檔裡實際的方法代號：卡片欄位消融沿用原方法的欄名，其餘與代號相同
def source(key):
    return CARD_FIELDS[key][2] if key in CARD_FIELDS else key
# 消融：(區塊, 比較對象, 改了什麼, 方法代號)。A 區除第一列外都已關掉作者加分（零件排名是在圖模型上算的）
ABLATION = [
    ("A", None, "Full main method", "adaptive_graph_creator"),
    ("A", "adaptive_graph_creator", "− familiar-author bonus (= graph model)", "adaptive"),
    ("A", "adaptive", "graph model − learned IDs", "adaptive_nolearn"),
    ("A", "adaptive", "graph model − text graph (learned IDs only)", "adaptive_notext"),
    ("A", "adaptive", "graph model − contrastive alignment (BPR-only IDs)", "adaptive_nocontrast"),
    ("A", "adaptive", "graph model − adaptive share (fixed share)", "fixed"),
    ("A", "adaptive", "graph model − popularity (personal score only)", "two_view"),
    ("A", "adaptive", "graph model − personalisation (730-day popularity only)", "two_year"),
    ("B", None, "Title only", "adaptive_title"),
    ("B", "adaptive_title", "+ ingredients and tags (full card)", "adaptive_card"),
    ("B", "adaptive_card", "+ interaction graph", "adaptive_nolearn"),
    # C 區：從整張卡拿掉一個欄位，在三個層級各比一次（混合權重不重調）
    ("C", "adaptive_card", "full-card profile − ingredients", "card_profile_no_ingredients"),
    ("C", "adaptive_card", "full-card profile − tags", "card_profile_no_tags"),
    ("C", "adaptive_nolearn", "card graph − ingredients", "card_graph_no_ingredients"),
    ("C", "adaptive_nolearn", "card graph − tags", "card_graph_no_tags"),
    ("C", "adaptive", "graph model − ingredients", "graph_model_no_ingredients"),
    ("C", "adaptive", "graph model − tags", "graph_model_no_tags"),
]


# 讀 results/ 底下的 JSON
def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# 併入一個方法的名次：同一位使用者在不同檔案裡必須是同一題
def merge_checked(base, part, name):
    part = part.rename(columns={"target_id": "check_target"})
    merged = base.merge(part, on="user_id", how="outer", validate="one_to_one")
    if merged.isna().any().any() or not merged.target_id.eq(merged.check_target).all():
        raise ValueError(f"{name}: users or targets differ from the semantic-graph ledger")
    return merged.drop(columns="check_target")


# 把搜尋格子裡某個設定試過的值寫成「a, b, c」（轉成 Python 原生型別，避免印出 np.float64）
def tried_values(series):
    return "{" + ", ".join(map(str, sorted(series.unique().tolist()))) + "}"


# 讀一個切點：以語意圖的題目表為基準併入各方法名次；檔案不存在的方法（使用者 kNN 沒有 2018 回測）略過
def load_split(split):
    base = pd.read_csv(RESULTS / f"semantic_graph/{split}_queries.csv",
                       usecols=["user_id", "target_id", "target_date", "candidate_count",
                                "training_support", "history_length"])
    author = pd.read_csv(RESULTS / f"creator_paths/{split}_graph_queries.csv",
                         usecols=["user_id", "target_id", "target_creator_prior_likes"])
    base = merge_checked(base, author, "creator_paths graph ledger")
    ledgers = {}
    for key, (_, _, prefix) in METHODS.items():
        path = RESULTS / (prefix.format(split=split) + "_queries.csv")
        if not path.exists():
            continue
        if path not in ledgers:
            ledgers[path] = pd.read_csv(path)
        column = "rank_" + source(key)
        part = ledgers[path][["user_id", "target_id", column]].rename(columns={column: "rank_" + key})
        base = merge_checked(base, part, path.name)
    return base


# 讀前十名清單（每個方法只取自己的結果檔），並核對「名次 ≤ 10 時清單該位置就是答案，否則答案不在清單裡」
def load_top10(split, table):
    lists = []
    for prefix in sorted({prefix for _, _, prefix in METHODS.values()}):
        path = RESULTS / (prefix.format(split=split) + "_top10.csv")
        if path.exists():
            frame = pd.read_csv(path)
            for key in [key for key, (_, _, p) in METHODS.items() if p == prefix]:
                lists.append(frame[frame.method == source(key)].assign(method=key))
    lists = pd.concat(lists, ignore_index=True)
    for key, group in lists.groupby("method"):
        merged = group.merge(table[["user_id", "target_id", "rank_" + key]], on="user_id", validate="one_to_one")
        if len(merged) != len(table):
            raise ValueError(f"{split}: {key} has lists for {len(merged)} of {len(table)} users")
        rank, target = merged["rank_" + key].to_numpy(), merged.target_id.to_numpy()
        matrix, hit = merged[ITEMS].to_numpy(), rank <= 10
        if not (matrix[hit, rank[hit] - 1] == target[hit]).all() or (matrix[~hit] == target[~hit, None]).any():
            raise ValueError(f"{split}: {key} ranks contradict its top-10 lists")
    return lists


# 一組名次的整體指標
def scores(rank):
    hit = rank <= 10
    return dict(hits=int(hit.sum()), hr_at_10=float(hit.mean()),
                ndcg_at_10=float(np.where(hit, 1 / np.log2(1 + rank), 0.0).mean()),
                median_rank=float(np.median(rank)))


# 一個方法在所有使用者的前十名裡，總共出現過幾道不同的食譜
def distinct(lists, key):
    return int(np.unique(lists.loc[lists.method == key, ITEMS].to_numpy()).size)


# 主實驗表：每個方法一列，2016 測試與 2018 回測各一組指標
def main_table(tables, lists):
    rows = []
    for key in MAIN_TABLE:
        label, family, _ = METHODS[key]
        row = dict(family=family, method=label, key=key)
        for split, table in tables.items():
            if "rank_" + key not in table:
                continue
            row.update({f"{split}_{name}": value for name, value in scores(table["rank_" + key].to_numpy()).items()})
            row[f"{split}_distinct_recipes"] = distinct(lists[split], key)
        rows.append(row)
    return pd.DataFrame(rows)


# 消融表：A 區和指定的上一層比，B 區和階梯的前一步比，C 區和原版卡片的同一層級比
def ablation_table(tables, lists):
    rows = []
    for block, reference, change, key in ABLATION:
        row = dict(block=block, change=change, method=METHODS[key][0], key=key,
                   compared_with=METHODS[reference][0] if reference else "")
        for split, table in tables.items():
            if "rank_" + key not in table:  # 卡片欄位消融尚未執行時留空
                continue
            now = scores(table["rank_" + key].to_numpy())
            row.update({f"{split}_hits": now["hits"], f"{split}_ndcg_at_10": now["ndcg_at_10"],
                        f"{split}_median_rank": now["median_rank"],
                        f"{split}_distinct_recipes": distinct(lists[split], key)})
            if reference:
                before = scores(table["rank_" + reference].to_numpy())
                row[f"{split}_delta_hits"] = now["hits"] - before["hits"]
                row[f"{split}_delta_ndcg"] = now["ndcg_at_10"] - before["ndcg_at_10"]
        rows.append(row)
    return pd.DataFrame(rows)


# 超參數：摘要表（每個設定一列）與完整搜尋紀錄（每個嘗試過的組合一列），全部以 2013 驗證集 NDCG@10 挑選
def hyperparameter_tables():
    rows, grid = [], []

    def record(family, setting, ndcg, selected):
        grid.append(dict(family=family, setting=setting, validation_ndcg_at_10=ndcg, selected=bool(selected)))

    # 熱門度的時間窗
    sel = read_json("validation_selection.json")
    for half, value in sel["half_life_validation"].items():
        record("Popularity half-life", f"half_life={half}", value, half == sel["half_life"])
    finite = max((h for h in sel["half_life_validation"] if h != "all"), key=sel["half_life_validation"].get)
    rows.append(dict(method="Popularity", parameter="half-life (days)",
                     tried=", ".join(sel["half_life_validation"]),
                     selected=f"{sel['half_life']} overall; {finite} = best finite window (Two-year popularity)",
                     validation_ndcg_at_10=sel["half_life_validation"][sel["half_life"]],
                     source="results/validation_selection.json"))

    # 食材 + 熱門度（同一張格子同時決定使用者 kNN 的比重）
    g = pd.read_csv(RESULTS / "validation_grid.csv")
    chosen = {"weighted": sel["ingredient_weight"], "raw": sel["raw_tuned_weight"]}
    for r in g.itertuples():
        record("Ingredients + popularity", f"pop_weight={r.pop_weight}, ingredient_weight={r.ingredient_weight}, "
               f"encoding={r.encoding}", r.ndcg10,
               r.pop_weight == sel["pop_weight"] and r.ingredient_weight == chosen[r.encoding])
    for encoding, label in (("weighted", "Rarity-weighted ingredients + popularity"),
                            ("raw", "Equal-weight ingredients + popularity")):
        best = g[(g.encoding == encoding) & (g.pop_weight == sel["pop_weight"])
                 & (g.ingredient_weight == chosen[encoding])].ndcg10.iloc[0]
        rows.append(dict(method=label, parameter="popularity share in base / ingredient weight",
                         tried=f"{tried_values(g.pop_weight)} × {tried_values(g.ingredient_weight)}",
                         selected=f"{sel['pop_weight']} / {chosen[encoding]}", validation_ndcg_at_10=best,
                         source="results/validation_grid.csv",
                         note="popularity share 1.0 means the user-kNN votes got weight 0 in the base"))

    # ALS
    a = pd.read_csv(RESULTS / "comparison/als_validation_grid.csv")
    als = read_json("comparison/als_selection.json")
    for r in a.itertuples():
        record("ALS", f"factors={r.factors}, confidence={r.confidence}, prior={r.prior}, prior_share={r.prior_share}",
               r.val_ndcg, any(r.factors == c["factors"] and r.confidence == c["confidence"]
                               and r.prior == c["prior"] and r.prior_share == c["prior_share"] for c in als.values()))
    for key, label in (("als", "ALS"), ("als_blend", "ALS + popularity")):
        c = als[key]
        rows.append(dict(method=label, parameter="factors / confidence / popularity window / popularity share",
                         tried=f"{tried_values(a.factors)} / {tried_values(a.confidence)} / "
                               f"{tried_values(a.prior)} / {tried_values(a.prior_share)}",
                         selected=f"{c['factors']} / {c['confidence']} / {c['prior']} / {c['prior_share']}",
                         validation_ndcg_at_10=c["validation_ndcg"], source="results/comparison/als_selection.json"))

    # 語意圖與熱門度的混合
    s = pd.read_csv(RESULTS / "semantic_graph/validation_grid.csv")
    graph = read_json("semantic_graph/selection.json")
    for r in s.itertuples():
        c = graph[r.method]
        record(f"Graph + popularity ({r.method})", f"prior={r.prior}, weight={r.weight}", r.ndcg,
               r.prior == c["prior"] and r.weight == c["weight"])
    for key, label in (("fixed", "Graph + popularity, fixed share"), ("adaptive", "Graph + popularity, adaptive share")):
        c = graph[key]
        rows.append(dict(method=label, parameter="popularity window / personal-score weight",
                         tried=f"{tried_values(s.prior)} / {tried_values(s.weight)}",
                         selected=f"{c['prior']} / {c['weight']}", validation_ndcg_at_10=c["ndcg"],
                         source="results/semantic_graph/selection.json",
                         note="the card / title / card-graph rows reuse this blend without re-tuning"))

    # 熟悉作者加分
    author = read_json("creator_paths/graph_selection.json")
    for weight, value in author["grid"]["adaptive_graph"].items():
        record("Familiar-author bonus", f"bonus={weight}", value, float(weight) == author["weights"]["adaptive_graph"])
    rows.append(dict(method="Adaptive graph + familiar author (main)", parameter="author bonus",
                     tried=", ".join(author["grid"]["adaptive_graph"]), selected=str(author["weights"]["adaptive_graph"]),
                     validation_ndcg_at_10=author["grid"]["adaptive_graph"][str(author["weights"]["adaptive_graph"])],
                     source="results/creator_paths/graph_selection.json"))

    # 聯合作者 ALS 的作者強度
    joint = read_json("author_als/selection.json")
    jm = pd.read_csv(RESULTS / "author_als/validation_metrics.csv")
    for r in jm.itertuples():
        record("Joint author ALS", f"strength={r.strength}", r.ndcg, r.strength == joint["strength"])
    rows.append(dict(method="Joint author ALS + popularity", parameter="author strength",
                     tried=", ".join(map(str, joint["strength_grid"])), selected=str(joint["strength"]),
                     validation_ndcg_at_10=float(jm.loc[jm.strength == joint["strength"], "ndcg"].iloc[0]),
                     source="results/author_als/selection.json",
                     note="validation picked 0 (= ALS + popularity); the main table reports strength 1.0 as a diagnostic"))

    # 圖模型 + ALS 的 ensemble 比重
    fusion = read_json("graph_fusion/selection.json")
    fm = pd.read_csv(RESULTS / "graph_fusion/validation_metrics.csv")
    for r in fm.itertuples():
        record("Graph + ALS fusion", f"als_share={r.als_share}", r.ndcg, r.als_share == fusion["als_share"])
    rows.append(dict(method="Graph + ALS fusion", parameter="ALS share",
                     tried=", ".join(map(str, fusion["share_grid"])), selected=str(fusion["als_share"]),
                     validation_ndcg_at_10=float(fm.loc[fm.als_share == fusion["als_share"], "ndcg"].iloc[0]),
                     source="results/graph_fusion/selection.json",
                     note="share 1.0 = pure ALS + popularity; the ensemble ranks equal ALS + popularity for every user"))

    # 沒有調、直接固定的設定
    rows.append(dict(method="Learned IDs (graph model)", parameter="dim / steps / learning rate / temperature / "
                     "alignment weight / BPR edges per step / seed",
                     tried="fixed, not tuned",
                     selected=f"{sg.DIM} / {sg.STEPS} / {sg.LEARNING_RATE} / {sg.TEMPERATURE} / {sg.CL_WEIGHT} / "
                              f"{sg.BPR_EDGES} / {sg.SEED}",
                     validation_ndcg_at_10=np.nan, source="experiments/semantic_graph.py"))
    rows.append(dict(method="Shared inputs", parameter="taste window / user-kNN neighbours / text encoder / card text",
                     tried="fixed, not tuned",
                     selected="last 20 likes / 80 / all-MiniLM-L6-v2 / title + ingredients + up to 6 non-generic tags",
                     validation_ndcg_at_10=np.nan,
                     source="experiments/run.py, semantic_graph.py, recipe_embeddings.py"))
    return pd.DataFrame(rows), pd.DataFrame(grid)


# 隨機種子：圖模型（不含作者加分）用三個種子重訓，報每個種子與平均 ± 標準差
def seed_tables():
    rows = []
    for split in SPLITS:
        base = pd.read_csv(RESULTS / f"semantic_graph/{split}_queries.csv", usecols=["user_id", "target_id", "rank_adaptive"])
        extra = pd.read_csv(RESULTS / f"semantic_graph/{split}_extra_seed_ranks.csv")
        merged = merge_checked(base, extra, f"{split} extra seeds")
        columns = {sg.SEED: "rank_adaptive"}
        columns.update({int(c.rsplit("_", 1)[1]): c for c in extra.columns if c.startswith("rank_seed_")})
        for seed, column in columns.items():
            rows.append(dict(split=split, seed=seed, **scores(merged[column].to_numpy())))
    frame = pd.DataFrame(rows)
    summary = frame.groupby("split", sort=False).agg(
        seeds=("seed", "size"), hits_mean=("hits", "mean"), hits_std=("hits", "std"),
        ndcg_mean=("ndcg_at_10", "mean"), ndcg_std=("ndcg_at_10", "std")).reset_index()
    return frame, summary


# 成本：離線建置秒數與每次推薦的毫秒數（2016 快照，basic-2 CPU；各程式的量測範圍不同，見 scope 欄）
def cost_table():
    log = (RESULTS / "logs/v1_pipeline.log")
    encode = {}
    if log.exists():
        for field, seconds in re.findall(r"^(card|title): 231,637/231,637, ([0-9.]+) s", log.read_text(), re.M):
            encode[field] = float(seconds)
    stats = read_json("test_stats.json")
    decay = read_json("finite_decay_summary.json")
    als = read_json("comparison/test_als_cost.json")["als"]
    joint = pd.read_csv(RESULTS / "author_als/test_metrics.csv").set_index("strength").loc[1.0]
    graph = read_json("semantic_graph/test_summary.json")["timing"]
    bench = read_json("semantic_graph/test_benchmark.json")["methods"]
    serving = read_json("creator_paths/serving.json")
    n = stats["n_questions"]
    return pd.DataFrame([
        dict(component="MiniLM encoding of every recipe card and title (one-off, shared by all text and graph methods)",
             offline_s=sum(encode.values()) if encode else np.nan, per_user_ms=np.nan,
             scope=f"card {encode.get('card', np.nan)} s + title {encode.get('title', np.nan)} s, 231,637 recipes"),
        dict(component="Popularity windows", offline_s=decay["index_build_s"], per_user_ms=decay["median_ms"],
             scope="median single-user ranking"),
        dict(component="Popularity, user-kNN and ingredient profiles (run.py, all together)",
             offline_s=sum(stats["build_s"].values()),
             per_user_ms=(stats["wall_s"] - sum(stats["build_s"].values())) / n * 1000,
             scope=f"average over {n} test users, all run.py scorers together"),
        dict(component="ALS (and ALS + popularity)", offline_s=als["build_s"], per_user_ms=als["batch_rank_ms_per_user"],
             scope="fit; batch ranking average"),
        dict(component="Joint author ALS (unit strength)", offline_s=joint.fit_s, per_user_ms=joint.score_s / n * 1000,
             scope="fit; batch scoring average"),
        dict(component="Card graph: build graph, project cards, propagate (no training)",
             offline_s=graph["snapshot_graph_s"] + graph["projection_and_history_s"] + graph["semantic_graph_s"],
             per_user_ms=bench["graph_text"]["median_ms"], scope="median warm single-user scoring incl. top-10"),
        dict(component="Learned IDs (BPR + alignment training)", offline_s=graph["training_s"],
             per_user_ms=bench["two_view"]["median_ms"], scope="per-user time is for the two-view score"),
        dict(component="Graph + popularity, adaptive share", offline_s=np.nan, per_user_ms=bench["adaptive"]["median_ms"],
             scope="median warm single-user scoring incl. top-10"),
        dict(component="Main method (adds the familiar-author bonus)", offline_s=np.nan, per_user_ms=serving["median_ms"],
             scope=f"median over {serving['queries']} single-user requests"),
    ])


# 食譜卡版本：每種卡片截斷後的平均 token 數、被 64 tokens 截斷的比例與編碼耗時（編碼時寫下的紀錄檔）
CARD_TEXTS = {
    "title": "title",
    "card": "title + ingredients + up to 6 tags",
    "card_no_ingredients": "title + up to 6 tags",
    "card_no_tags": "title + ingredients",
}


def card_variant_table():
    rows = []
    for field, content in CARD_TEXTS.items():
        path = RESULTS / f"recipe_embeddings/{field}.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text())
        rows.append(dict(card=field, text=content, mean_tokens=info["tokens"] / info["rows_done"],
                         truncated=info["truncated"], truncated_share=info["truncated"] / info["rows_done"],
                         encode_s=info["wall_seconds"]))
    return pd.DataFrame(rows)


# 匯出給第 3 部分：每位使用者的答案資訊與各方法名次，以及各方法的前十名清單
def export_predictions(split, table, lists, names):
    folder = OUT / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    out = table[["user_id", "target_id", "target_date", "history_length", "training_support",
                 "target_creator_prior_likes"]].rename(columns={
                     "training_support": "target_prior_likes", "target_creator_prior_likes": "user_likes_of_target_author"})
    out.insert(2, "target_name", out.target_id.map(names))
    out["familiar_author"] = out.user_likes_of_target_author > 0
    for key in METHODS:
        if "rank_" + key in table:
            out["rank_" + key] = table["rank_" + key]
    out.sort_values("user_id").to_csv(folder / f"{split}_ranks.csv", index=False)
    lists[["user_id", "method", *ITEMS]].sort_values(["method", "user_id"]).to_csv(folder / f"{split}_top10.csv", index=False)


# Markdown 表格：fmt 是每欄的格式函式；bold 欄位把數值最好的那列加粗
def markdown(frame, columns, bold=None):
    bold = bold or {}
    best = {col: (frame[col].max() if how == "max" else frame[col].min()) for col, how in bold.items()}
    lines = ["| " + " | ".join(title for _, title, _ in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col, _, fmt in columns:
            value = row.get(col)
            text = "—" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt(value)
            cells.append(f"**{text}**" if col in best and value == best[col] else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def integer(v):
    return f"{int(v):,}"


def percent(v):
    return f"{100 * v:.2f}%"


def gain(v):
    return f"{v:.5f}"


def signed(v):
    return f"{int(v):+d}"


def signed_gain(v):
    return f"{v:+.5f}"


def text(v):
    return str(v)


def seconds(v):
    return f"{v:.1f}"


def millis(v):
    return f"{v:.2f}"


# 把所有表格寫成一份 Markdown，方便貼進論文或交給寫手
def write_markdown(main, ablation, hyper, seeds, seed_summary, cost, cards, sizes):
    parts = ["# 第 2 部分：主實驗與消融實驗表格（自動產生）",
             f"由 `scripts/main_experiment_tables.py` 從 `results/` 產生；切點沿用 v1："
             f"2013 驗證選設定，2016 測試（{sizes['test']:,} 人），2018 回測（{sizes['prospective']:,} 人）。"
             "粗體為該欄最好的方法。"]
    for split, year in (("test", "2016 測試"), ("prospective", "2018 回測")):
        cols = [("family", "類別", text), ("method", "方法", text), (f"{split}_hits", "命中", integer),
                (f"{split}_hr_at_10", "HR@10", percent), (f"{split}_ndcg_at_10", "NDCG@10", gain),
                (f"{split}_distinct_recipes", "推出食譜數", integer), (f"{split}_median_rank", "中位名次", integer)]
        parts += [f"## 主實驗表（{year}）",
                  markdown(main, cols, {f"{split}_hits": "max", f"{split}_hr_at_10": "max",
                                        f"{split}_ndcg_at_10": "max", f"{split}_median_rank": "min"})]
    for split, year in (("test", "2016 測試"), ("prospective", "2018 回測")):
        cols = [("block", "區", text), ("change", "改了什麼", text), ("compared_with", "比較對象", text),
                (f"{split}_hits", "命中", integer), (f"{split}_delta_hits", "Δ命中", signed),
                (f"{split}_ndcg_at_10", "NDCG@10", gain), (f"{split}_delta_ndcg", "ΔNDCG@10", signed_gain),
                (f"{split}_distinct_recipes", "推出食譜數", integer), (f"{split}_median_rank", "中位名次", integer)]
        parts += [f"## 消融表（{year}）",
                  "A：主方法一次拿掉一個零件；B：文字特徵階梯；C：從整張卡拿掉一個欄位（混合權重都不重調）。",
                  markdown(ablation, cols)]
    parts += ["## 超參數（2013 驗證集，以 NDCG@10 挑選）",
              markdown(hyper, [("method", "方法", text), ("parameter", "設定", text), ("tried", "試過的值", text),
                               ("selected", "選中", text), ("validation_ndcg_at_10", "驗證 NDCG@10", gain),
                               ("note", "備註", text)])]
    parts += ["## 隨機種子（圖模型，不含作者加分）",
              markdown(seeds, [("split", "切點", text), ("seed", "種子", text), ("hits", "命中", integer),
                               ("ndcg_at_10", "NDCG@10", gain)]),
              markdown(seed_summary, [("split", "切點", text), ("seeds", "種子數", integer),
                                      ("hits_mean", "命中平均", seconds), ("hits_std", "命中標準差", seconds),
                                      ("ndcg_mean", "NDCG 平均", gain), ("ndcg_std", "NDCG 標準差", gain)])]
    parts += ["## 成本（2016 快照，basic-2 CPU）",
              markdown(cost, [("component", "元件", text), ("offline_s", "離線建置（秒）", seconds),
                              ("per_user_ms", "每次推薦（毫秒）", millis), ("scope", "量測範圍", text)])]
    parts += ["## 食譜卡版本（MiniLM 最多讀 64 tokens，超過的部分被截掉）",
              markdown(cards, [("card", "版本", text), ("text", "內容", text),
                               ("mean_tokens", "平均 token 數（截斷後）", seconds),
                               ("truncated", "被截斷的食譜數", integer), ("truncated_share", "截斷比例", percent),
                               ("encode_s", "編碼秒數", seconds)])]
    (OUT / "tables.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


# 主流程
def main():
    OUT.mkdir(exist_ok=True)
    names = pd.read_csv(DATA / "RAW_recipes.csv", usecols=["id", "name"]).set_index("id").name
    tables, lists = {}, {}
    for split in SPLITS:
        tables[split] = load_split(split)
        lists[split] = load_top10(split, tables[split])
        export_predictions(split, tables[split], lists[split], names)
        print(f"[{split}] {len(tables[split])} users, {lists[split].method.nunique()} methods", flush=True)

    main_rows = main_table(tables, lists)
    ablation = ablation_table(tables, lists)
    hyper, grid = hyperparameter_tables()
    seeds, seed_summary = seed_tables()
    cost = cost_table()
    cards = card_variant_table()
    for name, frame in (("main_table", main_rows), ("ablation_table", ablation), ("hyperparameters", hyper),
                        ("hyperparameter_grid", grid), ("seeds", seeds), ("seed_summary", seed_summary),
                        ("cost_table", cost), ("card_variants", cards)):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    pd.DataFrame([dict(key=k, method=label, family=family, source=prefix + "_queries.csv / _top10.csv",
                       in_main_table=k in MAIN_TABLE, in_ablation=any(k == a[3] for a in ABLATION))
                  for k, (label, family, prefix) in METHODS.items()]).to_csv(OUT / "methods.csv", index=False)
    write_markdown(main_rows, ablation, hyper, seeds, seed_summary, cost, cards,
                   {split: len(table) for split, table in tables.items()})
    print(main_rows[["method", "test_hits", "test_ndcg_at_10", "prospective_hits"]].to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
