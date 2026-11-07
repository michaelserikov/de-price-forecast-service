"""Stage and (only with --push) upload the service to a Hugging Face Docker Space.

    python scripts/deploy_hf.py                  # dry run: builds the staging folder and lists its files
    python scripts/deploy_hf.py --push           # creates the Space if needed and uploads (public by default)

The token is read from HF_TOKEN or ~/.hf_token. The Space gets the Dockerfile and card from
deploy/huggingface/, the package, the current model and the latest forecasts and monitoring files.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = "michaelserikov/de-price-forecast"
FILES = ["pyproject.toml", "src/dpfs", "models/latest", "forecasts", "results/backtest/metrics.csv"]


def stage(target: Path) -> list[Path]:
    for rel in FILES:
        src = ROOT / rel
        dst = target / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    shutil.copy2(ROOT / "deploy" / "huggingface" / "Dockerfile", target / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "huggingface" / "README.md", target / "README.md")
    return sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())


def token() -> str:
    t = os.environ.get("HF_TOKEN")
    if not t:
        t = (Path.home() / ".hf_token").read_text(encoding="utf-8").strip()
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--push", action="store_true", help="upload to the Space; without it only a dry run")
    ap.add_argument("--space", default=SPACE)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        size = sum((Path(tmp) / f).stat().st_size for f in files)
        print(f"{len(files)} files, {size / 1e6:.1f} MB")
        for f in files:
            print("  ", f.as_posix())
        if not args.push:
            print("dry run: nothing uploaded")
            return
        from huggingface_hub import HfApi
        api = HfApi(token=token())
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space", commit_message="Deploy service")
        print(f"uploaded: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
