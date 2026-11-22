# Food.com 全目錄食譜推薦

一位使用者下次會喜歡哪道食譜？本專案從 Food.com 的食譜卡與歷史評分重建推薦當日的目錄，為每位回訪使用者排列所有已發表、且自己尚未評過分的食譜。實驗比較熱門度、食材、協同過濾與語意圖模型，並檢驗使用者的口味一致性與對作者的熟悉度能否改善排名。

## 重現實驗

從儲存庫根目錄執行下列指令。論文使用 Apple M2 Pro（32 GB），數值運算與文字編碼各有一組依賴。以下指令適用於 macOS 或 Linux。

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.12 -m venv .venv-encoder
.venv-encoder/bin/python -m pip install -r requirements-encoder.txt
.venv/bin/python scripts/reproduce.py --check
```

這個指令會下載 Food.com 資料與指定版本的 MiniLM，核對 SHA-256、編碼食譜、訓練各模型，在兩個測試日期產生推薦，並比對表 1–3。原始 CSV 與文字向量位於 `data/`，訓練快取與實驗輸出位於 `results/`，均由程式建立。重建資料、快取與結果約需 1.7 GiB 磁碟空間，另計 Python 環境。論文的文字編碼使用 Apple Metal，其餘模型使用 CPU。執行器會自動選擇可用的編碼後端。若手邊已有原始 ZIP，或想指定 CPU 編碼，可在執行時指定壓縮檔、編碼環境與裝置。

```sh
.venv/bin/python scripts/reproduce.py --archive /path/to/food-com.zip \
  --encoder-python .venv-encoder/bin/python --device cpu
```

`--check` 要求命中數、食譜涵蓋數與案例名次完全相同，HR 與 NDCG 則比對至論文呈現的小數位。食譜向量的快取會記錄編碼後端。若要比較 CPU 與 Metal，可在兩份儲存庫副本中分別保留各自的快取。不同推論後端或數值函式庫可能改變極接近的分數與名次。程式會列出有差異的資料列與欄位，並保留實際產生的結果。跑新模型時可以省略 `--check`。

### 分階段執行

`--plan` 會列出各階段實際執行的命令。也可以指定一個或數個階段，重用上游已經完成的資料。以下指令在訓練完成後重算統計、圖與表。

```sh
.venv/bin/python scripts/reproduce.py --plan
.venv/bin/python scripts/reproduce.py evaluate tables --check
```

| 階段 | 內容 | 所需上游 |
| --- | --- | --- |
| `data` | 從 Kaggle 取得並驗證兩張原始 CSV，亦可使用本機 ZIP。 | 無 |
| `encode` | 驗證預訓練權重，分別將菜名、完整食譜卡編碼為向量。 | data |
| `baselines` | 訓練並驗證熱門度、食材、鄰居投票與隱式回饋 ALS 等基準。 | data |
| `graph` | 在每個歷史切點分別擬合語意圖、成對排序及對比學習，執行消融與不同種子的訓練。 | encode |
| `authors` | 驗證作者加分的權重，訓練聯合學習食譜與作者的 ALS，統計評估群體與重疊情形。 | baselines、graph |
| `fusion` | 以驗證集選擇語意圖與 ALS 的混合比例，核對混合端點與各自原始模型的排序。 | baselines、graph |
| `evaluate` | 檢查候選及前十名清單，計算配對信賴區間、作者熟悉度分組、案例、成本，以及圖 2–3。 | authors、fusion |
| `tables` | 從逐使用者的結果重建表 1–3，可與 `reference/` 核對。 | authors（包含 baselines、graph） |

## 評估設定

我們把四星和五星視為好評。使用者在預測日前至少要有五筆好評，並在觀察期間對符合候選條件的食譜留下好評，才會構成一筆評估。任何已評過分的食譜（包括零星）都會排除。目標是預測日後第一道獲得該使用者好評的合格食譜。同一天有多道時選編號最小者。所有方法排列同一組候選，分數相同時由食譜編號小者優先。

| 預測日 | 用途 | 觀察期間 | 已發表食譜 | 評估人數 |
| --- | --- | --- | ---: | ---: |
| 2013-01-01 | 選擇模型設定 | 2013–2015 | 225,809 | 4,031 |
| 2016-01-01 | 主要測試 | 2016–2018 | 230,956 | 1,480 |
| 2018-01-01 | 以相同設定重新訓練並回測 | 2018 | 231,448 | 532 |

食譜卡使用資料集保存的文字，沒有逐次修訂日期。MiniLM 的預訓練也晚於這些預測日。Food.com 的候選目錄、歷史評分、模型擬合與熱門度統計都依各自的預測日截斷。

兩個測試期間重疊，有 142 個相同的「使用者、目標食譜」組合。程式另外列出 2018 年其餘 390 個組合的成績。HR@10 衡量有多少使用者能在前十名找到目標，NDCG@10 同時考慮命中位置。涵蓋數是所有清單合計出現過的不同食譜數。

## 結果與圖表

| 路徑 | 內容 |
| --- | --- |
| [表 1](reference/table1_cohorts.csv)、[表 2](reference/table2_rankings.csv)、[表 3](reference/table3_cases.csv) | 論文的參考數值，分別列出評估人數、14 種方法的成績，以及 3 個案例的目標名次。 |
| `results/paper_tables/` | 本次執行重建的資料規模、整體指標與案例排名 CSV。 |
| `results/semantic_graph/` | 超參數選擇、訓練曲線、不同種子、消融結果，以及重算候選分數所需的快取。 |
| `results/creator_paths/`、`results/author_als/`、`results/graph_fusion/` | 作者加分、聯合分解、圖模型與 ALS 混合各自的驗證設定、目標排名與前十名清單。 |
| `results/comparison/` | ALS 基準、配對檢定、分組結果、案例，以及 `figures/` 下的兩張統計圖。 |
| `results/creator_paths/worked_queries.json` | 三個案例的食譜名稱、目標分數、作者加分與前十名變化。 |

圖 1 的架構圖原始碼在 [figures/pipeline.tex](figures/pipeline.tex)。它可獨立用 `pdflatex -output-directory=results figures/pipeline.tex` 編譯。Python 實驗流程不需要 LaTeX。

## 延伸一個推薦方法

[experiments/run.py](experiments/run.py) 提供資料清理、時間切分、候選過濾、同分排序與前十名擷取。[experiments/semantic_graph.py](experiments/semantic_graph.py) 實作語意圖與可訓練的識別碼向量。[experiments/creator_paths.py](experiments/creator_paths.py) 示範如何在相同候選食譜上加入新訊號，從 2013 年選權重，再將設定用於兩個測試日。

若要在既有圖模型上開發重排序器，完成 `graph` 階段後，可以直接讀取已凍結的計分器。以下會輸出 2016 年測試群體中一位使用者的編號與所推薦的十個食譜編號。

```sh
PYTHONPATH=experiments .venv/bin/python - <<'PYTHON'
from run import ranked_top
from semantic_graph import load_cached_scorer

scorer = load_cached_scorer("test")
scores = scorer.score_batch(0, 32)[0]
recipe_ids = scorer.candidate_ids[ranked_top(scores)]
print(int(scorer.user_ids[0]), recipe_ids.tolist())
PYTHON
```

計分矩陣的欄位順序對應 `scorer.candidate_ids`，已評過分的食譜會被標記為負無窮大。新增分數時請保留這個遮罩，並以 2013 年的資料選定權重，再固定設定評估兩個測試日。

若要比較自己的方法，請為每個評估日輸出兩份 CSV。第一份每位使用者一列，第二份每位使用者、每個方法一列。以下假設新方法叫 `my_model`。

- `test_queries.csv` 的必要欄位為 `user_id,target_id,candidate_count,rank_my_model`。目標名次從 1 起算，針對該使用者的完整候選目錄計算。
- `test_top10.csv` 的欄位為 `user_id,method,item_1,…,item_10`，其中 `method` 填 `my_model`，每個 `item` 欄位存 Food.com 原始食譜編號。

將這兩份檔案交給相同的評估程式，即可比較新模型與現有基準。以下以原本的自適應圖模型 `adaptive` 作為案例對照。

```sh
.venv/bin/python experiments/evaluation.py --split test --semantic \
  --primary my_model --case-control adaptive \
  --extra results/extensions/test_queries.csv results/extensions/test_top10.csv
```

## 資料與方法來源

- 使用者評分與食譜卡來自 Food.com 公開資料集。[Kaggle 資料集](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) · [Majumder 等人，EMNLP 2019](https://aclanthology.org/D19-1613/)
- 食譜文字與使用者互動圖的設計參考 TESMR。本實作用食譜卡文字、歷史評分與作者進行全目錄預測。[Shin 等人，SIGIR 2026](https://doi.org/10.1145/3805712.3809863)
- 文字編碼使用固定版本的 MiniLM，下載程式會核對權重與詞元化器的雜湊值。[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- 互動圖上的鄰居聚合與隱式回饋矩陣分解分別參考 [LightGCN](https://doi.org/10.1145/3397271.3401063) 和 [Hu 等人的信心加權 ALS](https://doi.org/10.1109/ICDM.2008.22)。
