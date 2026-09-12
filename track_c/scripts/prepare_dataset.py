"""One-time setup: fetch raw MovieLens ml-1m and convert it to the RecBole
atomic-file format (`ml-1m.inter` [, `.user`, `.item`]) under `dataset/`.

RecBole's own hosted mirror of pre-converted datasets
(recbole.s3-accelerate.amazonaws.com) currently returns 403 AccessDenied, so
`create_dataset()`'s automatic download can't be relied on - this script
builds the atomic files directly from the official GroupLens release
instead.

Only `ml-1m.inter` is actually required by configs/dataset.yaml (it doesn't
load user/item side features), but `.user`/`.item` are written too since
they're nearly free and make the raw data usable for other tracks/experiments
without re-deriving it.

Usage (from the repo root):
    python track_c/scripts/prepare_dataset.py

Downloads to a temp dir and writes to dataset/ml-1m/ (RecBole's default
data_path). Safe to re-run; skips the download if the zip is already cached.
"""

import argparse
import ssl
import urllib.request
import zipfile
from pathlib import Path

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "dataset"

# Expected raw-file line counts for the official ml-1m release - a cheap
# sanity check that we didn't download a truncated/corrupted/wrong file.
EXPECTED_LINE_COUNTS = {"ratings.dat": 1_000_209, "users.dat": 6_040, "movies.dat": 3_883}


def download_zip(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Using cached {dest}")
        return dest

    print(f"Downloading {ML_1M_URL} ...")
    try:
        urllib.request.urlretrieve(ML_1M_URL, dest)
    except Exception as exc:  # noqa: BLE001 - fall back below on any TLS failure
        # files.grouplens.org's TLS certificate has been observed expired
        # (independently reproducible with `curl -vI <url>` -> "certificate
        # has expired"). That's a hosting issue on GroupLens's side, not a
        # local trust-store problem - retry once without verification rather
        # than failing outright, since this is a well-known, integrity-checked
        # public research dataset (line counts verified below), not sensitive
        # data.
        print(f"  verified download failed ({exc}); retrying without TLS verification...")
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(ML_1M_URL, context=ctx) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def extract(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    raw_dir = out_dir / "ml-1m"
    for name, expected in EXPECTED_LINE_COUNTS.items():
        path = raw_dir / name
        n_lines = sum(1 for _ in open(path, encoding="latin-1"))
        if n_lines != expected:
            raise ValueError(
                f"{path} has {n_lines} lines, expected {expected} - "
                "download looks corrupted or truncated, delete the cached zip and retry."
            )
    return raw_dir


def write_inter(raw_dir: Path, out_dir: Path) -> None:
    out_path = out_dir / "ml-1m.inter"
    with open(raw_dir / "ratings.dat", encoding="latin-1") as fin, open(out_path, "w") as fout:
        fout.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for line in fin:
            user_id, item_id, rating, timestamp = line.strip().split("::")
            fout.write(f"{user_id}\t{item_id}\t{rating}\t{timestamp}\n")
    print(f"Wrote {out_path}")


def write_user(raw_dir: Path, out_dir: Path) -> None:
    out_path = out_dir / "ml-1m.user"
    with open(raw_dir / "users.dat", encoding="latin-1") as fin, open(out_path, "w") as fout:
        fout.write("user_id:token\tgender:token\tage:token\toccupation:token\tzip_code:token\n")
        for line in fin:
            fout.write("\t".join(line.strip().split("::")) + "\n")
    print(f"Wrote {out_path}")


def write_item(raw_dir: Path, out_dir: Path) -> None:
    out_path = out_dir / "ml-1m.item"
    with open(raw_dir / "movies.dat", encoding="latin-1") as fin, open(out_path, "w") as fout:
        fout.write("item_id:token\tmovie_title:token_seq\trelease_year:token\tgenre:token_seq\n")
        for line in fin:
            item_id, title, genres = line.strip().split("::")
            year = title.rstrip(")").rsplit("(", 1)[-1] if "(" in title else ""
            fout.write(f"{item_id}\t{title}\t{year}\t{' '.join(genres.split('|'))}\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="RecBole data_path (default matches configs/*.yaml)")
    parser.add_argument("--tmp-dir", default="/tmp/track_c_ml1m_download")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    out_dir = Path(args.data_path) / "ml-1m"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_zip(tmp_dir / "ml-1m.zip")
    raw_dir = extract(zip_path, tmp_dir)

    write_inter(raw_dir, out_dir)
    write_user(raw_dir, out_dir)
    write_item(raw_dir, out_dir)

    print(f"\nDone. RecBole dataset ready at: {out_dir}")


if __name__ == "__main__":
    main()
