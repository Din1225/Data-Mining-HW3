# 第 2 部分：主實驗與消融實驗表格（自動產生）

由 `scripts/main_experiment_tables.py` 從 `results/` 產生；切點沿用 v1：2013 驗證選設定，2016 測試（1,480 人），2018 回測（532 人）。粗體為該欄最好的方法。

## 主實驗表（2016 測試）

| 類別 | 方法 | 命中 | HR@10 | NDCG@10 | 推出食譜數 | 中位名次 |
|---|---|---|---|---|---|---|
| Non-personalised | All-time popularity | 20 | 1.35% | 0.00599 | 34 | 10,114 |
| Non-personalised | Two-year popularity (730-day half-life) | 30 | 2.03% | **0.01161** | 26 | 7,724 |
| Collaborative filtering | User-kNN (80 neighbours) | 13 | 0.88% | 0.00463 | 4,293 | 73,153 |
| Collaborative filtering | ALS | 26 | 1.76% | 0.00833 | 399 | 40,310 |
| Collaborative filtering | ALS + popularity | **33** | **2.23%** | 0.01084 | 98 | 7,447 |
| Collaborative filtering | Joint author ALS + popularity (unit strength) | 31 | 2.09% | 0.00924 | 91 | 7,596 |
| Content + popularity | Equal-weight ingredients + popularity | 30 | 2.03% | 0.01046 | 74 | 9,560 |
| Content + popularity | Rarity-weighted ingredients + popularity | 25 | 1.69% | 0.00931 | 37 | 9,020 |
| Content + popularity | Title profile + popularity | 27 | 1.82% | 0.00887 | 440 | 12,720 |
| Content + popularity | Full-card profile + popularity | 27 | 1.82% | 0.00968 | 355 | 12,853 |
| Semantic graph | Card graph + popularity | 25 | 1.69% | 0.00920 | 122 | 6,873 |
| Semantic graph | Two-view graph | 22 | 1.49% | 0.00713 | 398 | 14,014 |
| Semantic graph | Graph + popularity, fixed share | 26 | 1.76% | 0.00887 | 119 | 6,943 |
| Semantic graph | Graph + popularity, adaptive share | 27 | 1.82% | 0.00917 | 130 | 6,960 |
| Ensemble | Graph + ALS fusion | **33** | **2.23%** | 0.01084 | 98 | 7,447 |
| Main method | Adaptive graph + familiar author (main) | 26 | 1.76% | 0.00897 | 128 | **6,793** |

## 主實驗表（2018 回測）

| 類別 | 方法 | 命中 | HR@10 | NDCG@10 | 推出食譜數 | 中位名次 |
|---|---|---|---|---|---|---|
| Non-personalised | All-time popularity | 4 | 0.75% | 0.00288 | 27 | 12,679 |
| Non-personalised | Two-year popularity (730-day half-life) | 9 | 1.69% | 0.00790 | 21 | 8,679 |
| Collaborative filtering | User-kNN (80 neighbours) | — | — | — | — | — |
| Collaborative filtering | ALS | 4 | 0.75% | 0.00301 | 277 | 41,335 |
| Collaborative filtering | ALS + popularity | 9 | 1.69% | 0.00801 | 79 | 8,675 |
| Collaborative filtering | Joint author ALS + popularity (unit strength) | 8 | 1.50% | 0.00752 | 85 | 7,879 |
| Content + popularity | Equal-weight ingredients + popularity | 5 | 0.94% | 0.00523 | 55 | 12,246 |
| Content + popularity | Rarity-weighted ingredients + popularity | 5 | 0.94% | 0.00511 | 33 | 12,051 |
| Content + popularity | Title profile + popularity | 7 | 1.32% | 0.00552 | 365 | 18,609 |
| Content + popularity | Full-card profile + popularity | 9 | 1.69% | **0.00956** | 281 | 20,058 |
| Semantic graph | Card graph + popularity | **10** | **1.88%** | 0.00950 | 109 | **7,834** |
| Semantic graph | Two-view graph | 5 | 0.94% | 0.00382 | 255 | 19,321 |
| Semantic graph | Graph + popularity, fixed share | 6 | 1.13% | 0.00431 | 95 | 8,262 |
| Semantic graph | Graph + popularity, adaptive share | 7 | 1.32% | 0.00488 | 94 | 9,014 |
| Ensemble | Graph + ALS fusion | 9 | 1.69% | 0.00801 | 79 | 8,675 |
| Main method | Adaptive graph + familiar author (main) | 6 | 1.13% | 0.00437 | 96 | 7,988 |

## 消融表（2016 測試）

A：主方法一次拿掉一個零件；B：文字特徵階梯；C：從整張卡拿掉一個欄位（混合權重都不重調）。

| 區 | 改了什麼 | 比較對象 | 命中 | Δ命中 | NDCG@10 | ΔNDCG@10 | 推出食譜數 | 中位名次 |
|---|---|---|---|---|---|---|---|---|
| A | Full main method |  | 26 | — | 0.00897 | — | 128 | 6,793 |
| A | − familiar-author bonus (= graph model) | Adaptive graph + familiar author (main) | 27 | +1 | 0.00917 | +0.00019 | 130 | 6,960 |
| A | graph model − learned IDs | Graph + popularity, adaptive share | 25 | -2 | 0.00920 | +0.00003 | 122 | 6,873 |
| A | graph model − text graph (learned IDs only) | Graph + popularity, adaptive share | 20 | -7 | 0.00726 | -0.00191 | 138 | 7,412 |
| A | graph model − contrastive alignment (BPR-only IDs) | Graph + popularity, adaptive share | 30 | +3 | 0.00885 | -0.00031 | 114 | 7,111 |
| A | graph model − adaptive share (fixed share) | Graph + popularity, adaptive share | 26 | -1 | 0.00887 | -0.00030 | 119 | 6,943 |
| A | graph model − popularity (personal score only) | Graph + popularity, adaptive share | 22 | -5 | 0.00713 | -0.00203 | 398 | 14,014 |
| A | graph model − personalisation (730-day popularity only) | Graph + popularity, adaptive share | 30 | +3 | 0.01161 | +0.00245 | 26 | 7,724 |
| B | Title only |  | 27 | — | 0.00887 | — | 440 | 12,720 |
| B | + ingredients and tags (full card) | Title profile + popularity | 27 | +0 | 0.00968 | +0.00081 | 355 | 12,853 |
| B | + interaction graph | Full-card profile + popularity | 25 | -2 | 0.00920 | -0.00049 | 122 | 6,873 |
| C | full-card profile − ingredients | Full-card profile + popularity | 24 | -3 | 0.00939 | -0.00029 | 432 | 13,083 |
| C | full-card profile − tags | Full-card profile + popularity | 31 | +4 | 0.01066 | +0.00098 | 348 | 13,619 |
| C | card graph − ingredients | Card graph + popularity | 25 | +0 | 0.00989 | +0.00069 | 138 | 7,129 |
| C | card graph − tags | Card graph + popularity | 26 | +1 | 0.00876 | -0.00043 | 121 | 6,861 |
| C | graph model − ingredients | Graph + popularity, adaptive share | 27 | +0 | 0.00898 | -0.00018 | 121 | 7,348 |
| C | graph model − tags | Graph + popularity, adaptive share | 28 | +1 | 0.00931 | +0.00014 | 123 | 6,869 |

## 消融表（2018 回測）

A：主方法一次拿掉一個零件；B：文字特徵階梯；C：從整張卡拿掉一個欄位（混合權重都不重調）。

| 區 | 改了什麼 | 比較對象 | 命中 | Δ命中 | NDCG@10 | ΔNDCG@10 | 推出食譜數 | 中位名次 |
|---|---|---|---|---|---|---|---|---|
| A | Full main method |  | 6 | — | 0.00437 | — | 96 | 7,988 |
| A | − familiar-author bonus (= graph model) | Adaptive graph + familiar author (main) | 7 | +1 | 0.00488 | +0.00051 | 94 | 9,014 |
| A | graph model − learned IDs | Graph + popularity, adaptive share | 10 | +3 | 0.00950 | +0.00462 | 109 | 7,834 |
| A | graph model − text graph (learned IDs only) | Graph + popularity, adaptive share | 4 | -3 | 0.00284 | -0.00204 | 102 | 9,312 |
| A | graph model − contrastive alignment (BPR-only IDs) | Graph + popularity, adaptive share | 8 | +1 | 0.00542 | +0.00054 | 94 | 9,303 |
| A | graph model − adaptive share (fixed share) | Graph + popularity, adaptive share | 6 | -1 | 0.00431 | -0.00057 | 95 | 8,262 |
| A | graph model − popularity (personal score only) | Graph + popularity, adaptive share | 5 | -2 | 0.00382 | -0.00106 | 255 | 19,321 |
| A | graph model − personalisation (730-day popularity only) | Graph + popularity, adaptive share | 9 | +2 | 0.00790 | +0.00303 | 21 | 8,679 |
| B | Title only |  | 7 | — | 0.00552 | — | 365 | 18,609 |
| B | + ingredients and tags (full card) | Title profile + popularity | 9 | +2 | 0.00956 | +0.00403 | 281 | 20,058 |
| B | + interaction graph | Full-card profile + popularity | 10 | +1 | 0.00950 | -0.00006 | 109 | 7,834 |
| C | full-card profile − ingredients | Full-card profile + popularity | 8 | -1 | 0.00737 | -0.00219 | 328 | 19,554 |
| C | full-card profile − tags | Full-card profile + popularity | 9 | +0 | 0.00849 | -0.00107 | 278 | 20,978 |
| C | card graph − ingredients | Card graph + popularity | 9 | -1 | 0.00956 | +0.00006 | 111 | 7,904 |
| C | card graph − tags | Card graph + popularity | 10 | +0 | 0.00921 | -0.00029 | 107 | 7,963 |
| C | graph model − ingredients | Graph + popularity, adaptive share | 7 | +0 | 0.00470 | -0.00018 | 99 | 9,063 |
| C | graph model − tags | Graph + popularity, adaptive share | 8 | +1 | 0.00628 | +0.00140 | 99 | 8,959 |

## 超參數（2013 驗證集，以 NDCG@10 挑選）

| 方法 | 設定 | 試過的值 | 選中 | 驗證 NDCG@10 | 備註 |
|---|---|---|---|---|---|
| Popularity | half-life (days) | 180, 730, all | all overall; 730 = best finite window (Two-year popularity) | 0.00892 | — |
| Rarity-weighted ingredients + popularity | popularity share in base / ingredient weight | {0.0, 0.15, 0.35, 0.55, 0.75, 1.0} × {0.0, 0.1, 0.2, 0.35, 0.5, 0.65} | 1.0 / 0.2 | 0.00955 | popularity share 1.0 means the user-kNN votes got weight 0 in the base |
| Equal-weight ingredients + popularity | popularity share in base / ingredient weight | {0.0, 0.15, 0.35, 0.55, 0.75, 1.0} × {0.0, 0.1, 0.2, 0.35, 0.5, 0.65} | 1.0 / 0.35 | 0.00935 | popularity share 1.0 means the user-kNN votes got weight 0 in the base |
| ALS | factors / confidence / popularity window / popularity share | {32, 64} / {5, 30} / {730, all} / {0.0, 0.1, 0.25, 0.5, 0.75} | 64 / 5 / all / 0.0 | 0.00998 | — |
| ALS + popularity | factors / confidence / popularity window / popularity share | {32, 64} / {5, 30} / {730, all} / {0.0, 0.1, 0.25, 0.5, 0.75} | 64 / 5 / 730 / 0.75 | 0.01138 | — |
| Graph + popularity, fixed share | popularity window / personal-score weight | {180, 730, all} / {0.0, 0.1, 0.25, 0.5, 0.75, 1.0} | 730 / 0.5 | 0.00969 | the card / title / card-graph rows reuse this blend without re-tuning |
| Graph + popularity, adaptive share | popularity window / personal-score weight | {180, 730, all} / {0.0, 0.1, 0.25, 0.5, 0.75, 1.0} | 730 / 0.5 | 0.00992 | the card / title / card-graph rows reuse this blend without re-tuning |
| Adaptive graph + familiar author (main) | author bonus | 0.0, 0.025, 0.05, 0.1, 0.2, 0.4 | 0.05 | 0.01000 | — |
| Joint author ALS + popularity | author strength | 0.0, 0.25, 1.0, 4.0 | 0.0 | 0.01138 | validation picked 0 (= ALS + popularity); the main table reports strength 1.0 as a diagnostic |
| Graph + ALS fusion | ALS share | 0.0, 0.25, 0.5, 0.75, 1.0 | 1.0 | 0.01138 | share 1.0 = pure ALS + popularity; the ensemble ranks equal ALS + popularity for every user |
| Learned IDs (graph model) | dim / steps / learning rate / temperature / alignment weight / BPR edges per step / seed | fixed, not tuned | 64 / 80 / 0.012 / 0.2 / 0.05 / 32768 / 20261120 | — | — |
| Shared inputs | taste window / user-kNN neighbours / text encoder / card text | fixed, not tuned | last 20 likes / 80 / all-MiniLM-L6-v2 / title + ingredients + up to 6 non-generic tags | — | — |

## 隨機種子（圖模型，不含作者加分）

| 切點 | 種子 | 命中 | NDCG@10 |
|---|---|---|---|
| test | 20261120 | 27 | 0.00917 |
| test | 20261121 | 27 | 0.00874 |
| test | 20261122 | 30 | 0.00977 |
| prospective | 20261120 | 7 | 0.00488 |
| prospective | 20261121 | 5 | 0.00408 |
| prospective | 20261122 | 6 | 0.00548 |

| 切點 | 種子數 | 命中平均 | 命中標準差 | NDCG 平均 | NDCG 標準差 |
|---|---|---|---|---|---|
| test | 3 | 28.0 | 1.7 | 0.00923 | 0.00052 |
| prospective | 3 | 6.0 | 1.0 | 0.00481 | 0.00070 |

## 成本（2016 快照，basic-2 CPU）

| 元件 | 離線建置（秒） | 每次推薦（毫秒） | 量測範圍 |
|---|---|---|---|
| MiniLM encoding of every recipe card and title (one-off, shared by all text and graph methods) | 790.6 | — | card 684.0 s + title 106.6 s, 231,637 recipes |
| Popularity windows | 0.1 | 0.01 | median single-user ranking |
| Popularity, user-kNN and ingredient profiles (run.py, all together) | 10.2 | 20.14 | average over 1480 test users, all run.py scorers together |
| ALS (and ALS + popularity) | 18.3 | 2.77 | fit; batch ranking average |
| Joint author ALS (unit strength) | 21.7 | 2.18 | fit; batch scoring average |
| Card graph: build graph, project cards, propagate (no training) | 5.6 | 7.11 | median warm single-user scoring incl. top-10 |
| Learned IDs (BPR + alignment training) | 69.8 | 11.05 | per-user time is for the two-view score |
| Graph + popularity, adaptive share | — | 11.79 | median warm single-user scoring incl. top-10 |
| Main method (adds the familiar-author bonus) | — | 13.03 | median over 200 single-user requests |

## 食譜卡版本（MiniLM 最多讀 64 tokens，超過的部分被截掉）

| 版本 | 內容 | 平均 token 數（截斷後） | 被截斷的食譜數 | 截斷比例 | 編碼秒數 |
|---|---|---|---|---|---|
| title | title | 8.1 | 0 | 0.00% | 106.6 |
| card | title + ingredients + up to 6 tags | 57.9 | 92,737 | 40.04% | 684.0 |
| card_no_ingredients | title + up to 6 tags | 29.5 | 0 | 0.00% | 415.1 |
| card_no_tags | title + ingredients | 41.8 | 18,416 | 7.95% | 550.1 |
