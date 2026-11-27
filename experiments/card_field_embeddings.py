"""
用途：
    食譜卡欄位消融用的向量：把食譜卡拿掉一個欄位後，用與原版相同的 MiniLM 重新編碼。
    原版卡片 = 菜名 + 食材 + 最多 6 個標籤（recipe_embeddings.py 的 as_text），超過 64 tokens 會被截斷；
    這裡多做兩種卡片：card_no_ingredients（菜名 + 標籤）、card_no_tags（菜名 + 食材）。
    模型版本、截斷長度、批次與存檔方式都沿用原版程式，只換卡片文字。

主要流程：
    1. 讀食譜資料（路徑來自環境變數 FOODCOM_DATA_DIR），依編號排序，核對與原版向量的編號一致
    2. 載入原版下載好的 MiniLM checkpoint（不連網）
    3. 對每種卡片呼叫原版的 encode_field，寫到 data/cache/recipe_embeddings/<版本>.npy，
       並在 results/recipe_embeddings/<版本>.json 記錄耗時與被截斷的筆數

執行（repo 根目錄，需要 dm-hw3-enc 環境，只支援 CPU）：
    FOODCOM_DATA_DIR=data OMP_NUM_THREADS=16 conda run -n dm-hw3-enc python experiments/card_field_embeddings.py
"""

import argparse
import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import recipe_embeddings as base
from tokenizers import Tokenizer
from transformers.models.bert.modeling_bert import BertModel

VARIANTS = ("card_no_ingredients", "card_no_tags")
ORIGINAL_AS_TEXT = base.as_text


# 卡片文字：新版本只拿掉一個欄位，其餘格式與原版相同；原有的 card / title 交回原版函式
def as_text(row, field):
    if field not in VARIANTS:
        return ORIGINAL_AS_TEXT(row, field)
    title = row.name.strip() or "untitled recipe"
    if field == "card_no_tags":
        return f"{title}. Ingredients: {', '.join(literal_eval(row.ingredients))}."
    tags = [tag.replace("-", " ") for tag in literal_eval(row.tags) if tag not in base.GENERIC_TAGS]
    return f"{title}. Tags: {', '.join(tags[:6])}."


# 主流程：準備資料與模型，逐一編碼兩種卡片
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    # 原版 encode_field 透過模組內的全域名稱呼叫 as_text，替換後就會使用新的卡片文字
    base.as_text = as_text

    raw = Path(os.environ["FOODCOM_DATA_DIR"]) / "RAW_recipes.csv"
    catalogue = pd.read_csv(raw, usecols=["id", "name", "ingredients", "tags"], keep_default_na=False)
    catalogue = catalogue.sort_values("id").reset_index(drop=True)
    ids = np.load(base.CACHE / "ids.npy")
    if not np.array_equal(ids, catalogue.id.to_numpy(np.int64)):
        raise ValueError("Recipe IDs differ from the cached embeddings; run recipe_embeddings.py first")

    tokenizer = Tokenizer.from_file(str(base.CHECKPOINT / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=base.LENGTH)
    model = BertModel.from_pretrained(base.CHECKPOINT, local_files_only=True, attn_implementation="sdpa").eval()
    model.to("cpu")
    signature = dict(
        format_version=base.FORMAT_VERSION,
        model=base.MODEL_ID,
        revision=base.REVISION,
        token_limit=base.LENGTH,
        id_sha256=base.file_digest(base.CACHE / "ids.npy"),
        source_sha256=base.file_digest(raw),
        safetensor_sha256=base.file_digest(base.CHECKPOINT / "model.safetensors"),
        tokenizer_sha256=base.file_digest(base.CHECKPOINT / "tokenizer.json"),
        backend="cpu",
    )
    for field in args.fields:
        base.encode_field(catalogue, field, tokenizer, model, "cpu", signature)


if __name__ == "__main__":
    main()
