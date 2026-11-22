"""Download the public Food.com archive or use a local copy, then verify its two raw tables."""
import argparse
import hashlib
from http.client import HTTPException
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import zlib


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions"
ARCHIVE_NAME = "food-com.zip"
SIZES = {"RAW_recipes.csv": 294520189, "RAW_interactions.csv": 349436524}
CHUNK = 1024 * 1024


def expected_hashes():
    checksum_file = DATA / "SHA256SUMS"
    entries = {}
    for line in checksum_file.read_text().splitlines():
        if line.strip():
            digest, filename = line.split()
            entries[filename] = digest
    if set(entries) != set(SIZES):
        raise ValueError(f"Expected one SHA-256 per raw table in {checksum_file}")
    return entries


def verified(path, size, expected):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest() == expected


def unpack(archive, staging, missing, hashes):
    with ZipFile(archive) as package:
        for name in missing:
            records = [item for item in package.infolist() if item.filename == name]
            if len(records) != 1 or records[0].file_size != SIZES[name]:
                raise ValueError(f"{archive} lacks one correctly sized {name}")
            # Only literal, known filenames are written. ZIP member paths are never extracted.
            destination = staging / name
            with package.open(records[0]) as source, destination.open("wb") as output:
                digest = hashlib.sha256()
                while block := source.read(CHUNK):
                    output.write(block)
                    digest.update(block)
            if destination.stat().st_size != SIZES[name] or digest.hexdigest() != hashes[name]:
                raise ValueError(f"SHA-256 verification failed for {name} in {archive}")


def download(destination):
    print("Downloading the public Food.com dataset from Kaggle.", flush=True)
    request = Request(URL, headers={"User-Agent": "Foodcom-ranking-reproduction/1.0"})
    with urlopen(request, timeout=120) as source, destination.open("wb") as output:
        shutil.copyfileobj(source, output, length=CHUNK)


def prepare(data_dir, archive):
    hashes = expected_hashes()
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name, size in SIZES.items() if not verified(data_dir / name, size, hashes[name])]
    if not missing:
        print(f"Both raw tables match their published checksums in {data_dir}.")
        return
    cached = data_dir / ARCHIVE_NAME
    with TemporaryDirectory(prefix=".prepare-", dir=data_dir) as temporary:
        staging = Path(temporary)
        if archive is not None:
            # An explicitly supplied local file is never replaced or downloaded.
            unpack(archive, staging, missing, hashes)
        else:
            use_download = True
            if cached.exists():
                try:
                    unpack(cached, staging, missing, hashes)
                    use_download = False
                except (BadZipFile, EOFError, ValueError, zlib.error) as exc:
                    print(f"Cached archive failed verification ({exc}); requesting a fresh copy.", flush=True)
            if use_download:
                downloaded = staging / ARCHIVE_NAME
                download(downloaded)
                unpack(downloaded, staging, missing, hashes)
                downloaded.replace(cached)
        # Commit only after all required entries pass validation; retries reuse any valid table.
        for name in missing:
            (staging / name).replace(data_dir / name)
    print(f"Verified both raw tables in {data_dir}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA, help="Directory for the archive and raw tables")
    parser.add_argument("--archive", type=Path, help="Read this local ZIP rather than downloading from Kaggle")
    args = parser.parse_args()
    try:
        prepare(args.data_dir.expanduser(), args.archive.expanduser() if args.archive else None)
    except (OSError, EOFError, BadZipFile, ValueError, URLError, HTTPException, zlib.error) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
