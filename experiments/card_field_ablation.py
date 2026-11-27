"""
用途：
    食譜卡欄位消融：把圖模型使用的「整張卡」向量換成拿掉某個欄位的版本，其餘完全不變，
    重跑 2016 測試與 2018 回測。混合權重沿用原版在 2013 驗證集選出的設定
    （results/semantic_graph/selection.json），不重新調參數，與其他消融版本的做法一致。

主要流程：
    1. 讀資料（路徑來自環境變數 FOODCOM_DATA_DIR）
    2. 對每個卡片版本：把 semantic_graph 的輸出資料夾改到 results/card_fields/<版本>/，並複製原版選擇檔
    3. 讓 semantic_graph 讀「整張卡」時改讀該版本的向量，再呼叫原版的 run_snapshot
       （卡片向量同時影響：卡片口味、互動圖傳遞、學習 ID 的對齊目標、口味一致度閘門）
    4. 版本 card（原版向量）是對照組：名次必須和 results/semantic_graph/ 完全相同，
       證明這種替換方式沒有改到其他東西

執行（repo 根目錄，dm-hw3 環境；數值函式庫固定單執行緒，與 reproduce.py 相同）：
    FOODCOM_DATA_DIR=data OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        conda run -n dm-hw3 python experiments/card_field_ablation.py --variants card
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import semantic_graph as sg
from run import ROOT, load_data

OUT = ROOT / "results/card_fields"
ORIGINAL = ROOT / "results/semantic_graph"
ORIGINAL_PROJECT_CARDS = sg.project_cards


# 用指定的卡片向量跑一個版本；run_snapshot 透過模組全域名稱讀 OUTPUT 與 project_cards，所以直接替換
def run_variant(variant, splits, recipes, events):
    output = OUT / variant
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORIGINAL / "selection.json", output / "selection.json")
    sg.OUTPUT = output
    sg.project_cards = lambda kind, ids: ORIGINAL_PROJECT_CARDS(variant if kind == "card" else kind, ids)
    for split in splits:
        sg.run_snapshot(split, recipes, events)


# 對照組檢查：用原版卡片重跑的名次必須與原版結果逐人相同
def check_control(splits):
    for split in splits:
        new = pd.read_csv(OUT / "card" / f"{split}_queries.csv")
        old = pd.read_csv(ORIGINAL / f"{split}_queries.csv")
        columns = ["user_id", "target_id", *[c for c in old.columns if c.startswith("rank_")]]
        if not new[columns].equals(old[columns]):
            raise ValueError(f"Control run differs from {ORIGINAL}/{split}_queries.csv")
        print(f"control {split}: ranks identical to results/semantic_graph", flush=True)


# 主流程
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["card_no_ingredients", "card_no_tags"],
                        help="card = control with the original vectors")
    parser.add_argument("--splits", nargs="+", choices=("test", "prospective"), default=["test", "prospective"])
    args = parser.parse_args()
    recipes, events, _ = load_data(Path(os.environ["FOODCOM_DATA_DIR"]))
    for variant in args.variants:
        run_variant(variant, args.splits, recipes, events)
        if variant == "card":
            check_control(args.splits)


if __name__ == "__main__":
    main()
