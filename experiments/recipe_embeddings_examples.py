"""Inspect two preselected pairs from the 2016 recipe catalogue."""

import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/recipe_embeddings"
PAIRS = [(400104, 404343), (48155, 177847)]


def cosine(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    recipes = pd.read_csv(
        ROOT / "data/RAW_recipes.csv", usecols=["id", "name", "ingredients", "submitted"]
    )
    recipes = recipes.set_index("id")
    ids = np.load(CACHE / "ids.npy")
    matrices = {name: np.load(CACHE / f"{name}.npy", mmap_mode="r") for name in ("title", "card")}
    examples = []
    for pair in PAIRS:
        index = np.searchsorted(ids, pair)
        np.testing.assert_array_equal(ids[index], pair)
        rows = recipes.loc[list(pair)]
        assert (rows.submitted < "2016-01-01").all()
        ingredients = [literal_eval(value) for value in rows.ingredients]
        lexical_cosine = len(set(ingredients[0]) & set(ingredients[1])) / np.sqrt(
            len(set(ingredients[0])) * len(set(ingredients[1]))
        )
        examples.append(
            dict(
                recipe_ids=pair,
                names=rows.name.tolist(),
                ingredients=ingredients,
                binary_ingredient_cosine=float(lexical_cosine),
                **{
                    f"{key}_cosine": cosine(mat[index[0]], mat[index[1]])
                    for key, mat in matrices.items()
                },
            )
        )
    output = ROOT / "results/recipe_embeddings/examples.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(examples, indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
