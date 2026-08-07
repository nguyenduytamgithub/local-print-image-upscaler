from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = (
    (
        "facebook/sam2.1-hiera-base-plus",
        "b7320756a13354e7530a63935656d35b2f91a290",
        "2012733a0de5d03efd1bba550a2847c4551be9ef2e0d497c83074df66189f780",
    ),
    (
        "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
    ),
    (
        "hustvl/vitmatte-small-composition-1k",
        "53222614392e8bd24ed804fbd2f9a43c46ac3850",
        "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb",
    ),
)
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--lama", type=Path, required=True)
    args = parser.parse_args()
    if not args.lama.is_file() or sha256_file(args.lama).lower() != LAMA_SHA256:
        raise SystemExit(f"LaMa checkpoint missing or SHA-256 mismatch: {args.lama}")
    for repo_id, revision, weight_sha256 in MODELS:
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_files_only=args.check_only,
                    allow_patterns=("*.json", "*.safetensors", "vocab.txt"),
                )
            )
        except Exception as exc:
            mode = "local cache check" if args.check_only else "download"
            raise SystemExit(f"{repo_id}@{revision}: {mode} failed: {exc}") from exc
        weights = snapshot / "model.safetensors"
        if not weights.is_file():
            raise SystemExit(f"Pinned model weights are missing: {weights}")
        actual_sha256 = sha256_file(weights).lower()
        if actual_sha256 != weight_sha256:
            raise SystemExit(
                f"SHA-256 mismatch for {repo_id}@{revision}/model.safetensors: "
                f"expected {weight_sha256}, got {actual_sha256}"
            )
        print(f"OK: {repo_id}@{revision} (model.safetensors SHA-256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
