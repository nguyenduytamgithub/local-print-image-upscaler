"""Download revision-pinned V7 language assets and verify every runtime model."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import re
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
REQUIREMENTS_LOCK = HERE / "requirements.lock"
HASH_MANIFEST = HERE / "MODEL_SHA256SUMS.txt"
HF_REPO = "nrl-ai/vn-spell-correction-base"
HF_REVISION = "61596a71696ba360ae828f9db3806610afedf6d3"
HF_FILES = {
    "config.json": "ffd95f81f1665bd7dba68bccb1cc88f10787f9abe355af8a373f2d0f2eac5430",
    "generation_config.json": "eaaf16b98e47131e3ce155054140429e36c4ed6a23c6ff32c2946ef66ce121ff",
    "model.safetensors": "4275c526688289d06a054fd911d3ecebdad4a790707a371cf67a511f2251524c",
    "special_tokens_map.json": "1bf246679a41f9204ef9085d6ebe314cb3ec1c916db54e2a3be6e49c9c839453",
    "spiece.model": "59986b62f9f0b90edafb9b073ea7b93d21114a5841219a1ea2399ade73f729c6",
    "tokenizer.json": "637fb80d3a85ec307efeA089e10a42e1fa22fdbd19abe612e38cb771b78de8fd".lower(),
    "tokenizer_config.json": "10fe5aebb65886903db4ddb439d2192e28fc44db7cd87f988377a5d3f913cb97",
}
PADDLE_FILES = {
    "PP-OCRv6_medium_det/inference.json": "0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5",
    "PP-OCRv6_medium_det/inference.pdiparams": "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960",
    "PP-OCRv6_medium_det/inference.yml": "7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
    "PP-OCRv6_medium_rec/inference.json": "0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8",
    "PP-OCRv6_medium_rec/inference.pdiparams": "1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319",
    "PP-OCRv6_medium_rec/inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
}
TESSDATA_FILES = {
    "tessdata/vie.traineddata": "b6b49293d95d0b6dbd8780174627e82c75be957b6f4ed9862155540d6b00bb45",
}
_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)$")
_HASH_LINE_RE = re.compile(r"^([0-9A-Fa-f]{64})\s{2,}(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, expected: dict[str, str], label: str) -> None:
    failures = []
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {path}")
        else:
            actual = sha256(path)
            if actual.lower() != wanted.lower():
                failures.append(f"SHA-256 mismatch {path}: {actual}")
    if failures:
        raise RuntimeError(f"{label} verification failed:\n  - " + "\n  - ".join(failures))


def canonical_package_name(value: str) -> str:
    """PEP 503 normalization without importing a third-party package."""

    return re.sub(r"[-_.]+", "-", value).lower()


def read_requirements_lock(path: Path = REQUIREMENTS_LOCK) -> dict[str, tuple[str, str]]:
    pins: dict[str, tuple[str, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"requirements.lock:{line_number} is not an exact, marker-free pin: {line}"
            )
        display_name, version = match.groups()
        normalized = canonical_package_name(display_name)
        if normalized in pins:
            raise RuntimeError(f"duplicate package pin in requirements.lock: {display_name}")
        pins[normalized] = (display_name, version)
    if not pins:
        raise RuntimeError("requirements.lock contains no package pins")
    return pins


def verify_runtime_lock(path: Path = REQUIREMENTS_LOCK) -> None:
    """Prove that the isolated environment is exactly the committed lock."""

    pins = read_requirements_lock(path)
    installed: dict[str, tuple[str, str]] = {}
    duplicates: list[str] = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = canonical_package_name(name)
        if normalized in installed:
            duplicates.append(name)
        installed[normalized] = (name, distribution.version)

    failures: list[str] = []
    for normalized, (display_name, wanted) in pins.items():
        found = installed.get(normalized)
        if found is None:
            failures.append(f"missing {display_name}=={wanted}")
        elif found[1] != wanted:
            failures.append(f"{display_name}: expected {wanted}, installed {found[1]}")

    extras = sorted(set(installed) - set(pins))
    failures.extend(
        f"unlocked package {installed[name][0]}=={installed[name][1]}" for name in extras
    )
    failures.extend(f"duplicate installed distribution {name}" for name in duplicates)
    if failures:
        raise RuntimeError(
            "V7 runtime lock verification failed:\n  - " + "\n  - ".join(failures)
        )
    print(f"V7 package lock: verified ({len(pins)} exact distributions)")


def read_hash_manifest(path: Path = HASH_MANIFEST) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _HASH_LINE_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"MODEL_SHA256SUMS.txt:{line_number} is malformed")
        digest, relative = match.groups()
        normalized_path = relative.replace("\\", "/")
        manifest_path = Path(normalized_path)
        if manifest_path.is_absolute() or ".." in manifest_path.parts:
            raise RuntimeError(
                f"MODEL_SHA256SUMS.txt:{line_number} contains an unsafe path"
            )
        if normalized_path in entries:
            raise RuntimeError(f"duplicate model hash entry: {normalized_path}")
        entries[normalized_path] = digest.lower()
    return entries


def verify_hash_manifest_contract() -> None:
    """Keep the human-auditable manifest and executable constants in sync."""

    expected: dict[str, str] = {
        f"paddlex/official_models/{relative}": digest.lower()
        for relative, digest in PADDLE_FILES.items()
    }
    snapshot_prefix = (
        f"huggingface/models--{HF_REPO.replace('/', '--')}/snapshots/{HF_REVISION}"
    )
    expected.update(
        {f"{snapshot_prefix}/{relative}": digest.lower() for relative, digest in HF_FILES.items()}
    )
    expected.update(TESSDATA_FILES)
    manifest = read_hash_manifest()
    if manifest != expected:
        missing = sorted(set(expected) - set(manifest))
        extra = sorted(set(manifest) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(manifest)
            if expected[path] != manifest[path]
        )
        raise RuntimeError(
            "MODEL_SHA256SUMS.txt disagrees with model_setup.py: "
            + json.dumps(
                {"missing": missing, "extra": extra, "hash_mismatch": changed},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    print(f"V7 model hash manifest: verified ({len(expected)} pinned files)")


def language_snapshot(*, download: bool) -> Path:
    cache = MODELS / "huggingface"
    expected_snapshot = (
        cache / f"models--{HF_REPO.replace('/', '--')}" / "snapshots" / HF_REVISION
    )
    if download and not expected_snapshot.is_dir():
        from huggingface_hub import snapshot_download

        resolved = Path(
            snapshot_download(
                repo_id=HF_REPO,
                revision=HF_REVISION,
                cache_dir=cache,
                allow_patterns=sorted(HF_FILES),
            )
        )
        if resolved.resolve() != expected_snapshot.resolve():
            expected_snapshot.parent.mkdir(parents=True, exist_ok=True)
            if expected_snapshot.exists():
                raise RuntimeError(f"Unexpected existing snapshot: {expected_snapshot}")
            shutil.copytree(resolved, expected_snapshot)
    verify(expected_snapshot, HF_FILES, "Vietnamese suggestion model")
    return expected_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-runtime-only", action="store_true")
    parser.add_argument("--skip-language-model", action="store_true")
    args = parser.parse_args()
    verify_runtime_lock()
    verify_hash_manifest_contract()
    if args.check_runtime_only:
        return 0
    paddle_root = MODELS / "paddlex" / "official_models"
    verify(paddle_root, PADDLE_FILES, "PP-OCRv6")
    print("PP-OCRv6 medium: verified")
    if not args.skip_language_model:
        snapshot = language_snapshot(download=not args.check_only)
        print(f"Vietnamese suggestion model: verified ({snapshot})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
