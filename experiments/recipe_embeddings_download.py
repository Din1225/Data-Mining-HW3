"""Download the pinned public MiniLM encoder's tokenizer and portable model files."""

import shutil
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}"
CHECKPOINT = ROOT / "data/cache/recipe_embeddings/checkpoint"
FILES = {
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
}


def main():
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = CHECKPOINT / name
        if dest.is_file():
            with dest.open("rb") as cached:
                if file_digest(cached, "sha256").hexdigest() == expected:
                    print(name, expected, flush=True)
                    continue

        print(f"Downloading {name}", flush=True)
        with NamedTemporaryFile(dir=CHECKPOINT, prefix=f".{name}.", delete=False) as partial:
            pending = Path(partial.name)
        try:
            with urlopen(f"{BASE}/{name}", timeout=180) as source, pending.open("wb") as out:
                shutil.copyfileobj(source, out)
            with pending.open("rb") as verified:
                digest = file_digest(verified, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"Checksum mismatch for {name}: {digest}")
            pending.replace(dest)
            print(name, digest, flush=True)
        finally:
            pending.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
