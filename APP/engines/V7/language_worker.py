"""Offline JSONL worker for non-binding Vietnamese correction proposals.

Protocol (one JSON object per line)::

    {"id":"r1", "op":"propose", "text":"Toi yu Vit Nam",
     "brands":["TOP GIA"], "addresses":[]}

The response contains a proposal only.  This process has no operation that can
approve or apply text to an image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, TextIO


# These must be set before importing Transformers/Hugging Face.  The worker is
# intentionally a local-only process; a missing model is an error, not a reason
# to contact a service or download a replacement.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from v7lib.textnorm import (  # noqa: E402
    build_conservative_proposal,
    find_protected_spans,
    normalize_nfc,
)
from v7lib.types import LanguageProposal, ProtectedSpan  # noqa: E402


MODEL_CACHE_NAME = "models--nrl-ai--vn-spell-correction-base"
MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
)
MAX_REQUEST_CHARS = 20_000


class WorkerProtocolError(ValueError):
    """A stable, user-safe JSONL request error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _validate_model_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"local language model directory not found: {resolved}")
    missing = [name for name in MODEL_REQUIRED_FILES if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"local language model is incomplete; missing: {', '.join(missing)}"
        )
    return resolved


def resolve_local_model_dir(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve one complete local snapshot without consulting a model hub."""

    if value is not None:
        return _validate_model_dir(Path(value))
    configured = os.environ.get("V7_LANGUAGE_MODEL_DIR")
    if configured:
        return _validate_model_dir(Path(configured))

    snapshots = (
        Path(__file__).resolve().parent
        / "models"
        / "huggingface"
        / MODEL_CACHE_NAME
        / "snapshots"
    )
    candidates = sorted(
        path
        for path in snapshots.glob("*")
        if path.is_dir()
        and all((path / name).is_file() for name in MODEL_REQUIRED_FILES)
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "no complete local nrl-ai/vn-spell-correction-base snapshot was found"
        )
    raise RuntimeError(
        "multiple local language snapshots found; pass --model-dir to pin one"
    )


def _load_special_tokens(model_dir: Path) -> dict[str, Any]:
    special_path = model_dir / "special_tokens_map.json"
    if not special_path.is_file():
        return {"pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"}
    data = json.loads(special_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for key in ("pad_token", "eos_token", "unk_token", "bos_token", "sep_token"):
        value = data.get(key)
        if isinstance(value, str):
            result[key] = value
        elif isinstance(value, Mapping) and isinstance(value.get("content"), str):
            result[key] = value["content"]
    additional = data.get("additional_special_tokens")
    if isinstance(additional, list):
        result["additional_special_tokens"] = [
            item["content"] if isinstance(item, Mapping) else str(item)
            for item in additional
        ]
    result.setdefault("pad_token", "<pad>")
    result.setdefault("eos_token", "</s>")
    result.setdefault("unk_token", "<unk>")
    return result


def _choose_device(torch_module: Any, requested: str) -> str:
    value = requested.strip().lower()
    if value == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value != "cpu" and not re.fullmatch(r"cuda:\d+", value):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    return value


def _edge_whitespace(text: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(\s*)(.*?)(\s*)", text, flags=re.DOTALL)
    if match is None:
        return "", text, ""
    return match.group(1), match.group(2), match.group(3)


class LocalSpellModel:
    """One locally loaded ViT5 spelling model; network access is disabled."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        if not 16 <= int(max_length) <= 512:
            raise ValueError("max_length must be between 16 and 512")
        self.model_dir = resolve_local_model_dir(model_dir)
        self.max_length = int(max_length)

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
        except ImportError as exc:
            raise RuntimeError(
                "the V7 language worker requires local torch and transformers installs"
            ) from exc

        self._torch = torch
        self.device = _choose_device(torch, device)

        # AutoTokenizer currently fails on this model's exported metadata under
        # the pinned Transformers build.  Loading tokenizer.json directly is the
        # audited workaround; it also avoids trust_remote_code entirely.
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(self.model_dir / "tokenizer.json"),
            model_max_length=self.max_length,
            **_load_special_tokens(self.model_dir),
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_name = f"nrl-ai/vn-spell-correction-base@{self.model_dir.name}"

    def _generation_confidence(self, result: Any) -> float:
        scores = tuple(getattr(result, "scores", ()) or ())
        if not scores:
            return 0.0
        sequence = result.sequences[0]
        token_ids = sequence[-len(scores) :]
        log_probabilities: list[float] = []
        for step_scores, token_id in zip(scores, token_ids):
            token = int(token_id.item())
            if token in {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            }:
                continue
            log_probs = self._torch.log_softmax(step_scores[0].float(), dim=-1)
            log_probabilities.append(float(log_probs[token].item()))
        if not log_probabilities:
            return 0.0
        value = math.exp(sum(log_probabilities) / len(log_probabilities))
        return max(0.0, min(1.0, value))

    def _infer_unprotected(self, text: str, *, max_length: int) -> tuple[str, float, str | None]:
        prefix, core, suffix = _edge_whitespace(text)
        if not core or not any(character.isalpha() for character in core):
            return text, 1.0, None
        encoded = self.tokenizer(core, return_tensors="pt", truncation=False)
        input_length = int(encoded["input_ids"].shape[-1])
        if input_length > max_length:
            return text, 0.0, "unprotected-segment-too-long"
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            result = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated = self.tokenizer.decode(
            result.sequences[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generated = normalize_nfc(generated).strip()
        if not generated:
            return text, 0.0, "empty-model-output"
        return prefix + generated + suffix, self._generation_confidence(result), None

    def propose(
        self,
        text: str,
        *,
        brands: Iterable[str] = (),
        addresses: Iterable[str] = (),
        max_length: int | None = None,
    ) -> LanguageProposal:
        """Return a review-only proposal while preserving every critical span."""

        source = normalize_nfc(text)
        limit = self.max_length if max_length is None else int(max_length)
        if not 16 <= limit <= 512:
            raise WorkerProtocolError("invalid-max-length", "max_length must be 16..512")
        spans = find_protected_spans(source, brands=brands, addresses=addresses)

        # Never expose locked content to the generator.  Correct only the gaps
        # and stitch the original values back verbatim.  This is less fluent
        # than unrestricted generation, but is the required safety property for
        # prices, identifiers and customer-facing contact details.
        parts: list[str] = []
        confidences: list[float] = []
        reasons: list[str] = ["protected-spans-excluded-from-model"] if spans else []
        cursor = 0
        for span in (*spans, ProtectedSpan(len(source), len(source), "", "sentinel")):
            gap = source[cursor : span.start]
            corrected, confidence, warning = self._infer_unprotected(gap, max_length=limit)
            parts.append(corrected)
            if any(character.isalpha() for character in gap):
                confidences.append(confidence)
            if warning:
                reasons.append(warning)
            if span.kind != "sentinel":
                parts.append(span.text)
            cursor = span.end

        candidate = normalize_nfc("".join(parts))
        confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return build_conservative_proposal(
            source,
            candidate,
            confidence=confidence,
            brands=brands,
            addresses=addresses,
            protected_spans=spans,
            reasons=reasons,
            model=self.model_name,
        )


def _string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise WorkerProtocolError("invalid-request", f"{field} must be a list of strings")
    return [item for item in value if item.strip()]


def process_request(request: Mapping[str, Any], proposer: LocalSpellModel) -> dict[str, Any]:
    """Process one decoded JSON object; useful for both JSONL and unit tests."""

    request_id = request.get("id")
    operation = str(request.get("op", "propose"))
    if operation == "health":
        return {
            "id": request_id,
            "ok": True,
            "status": "ready",
            "offline": True,
            "model": proposer.model_name,
            "device": proposer.device,
        }
    if operation != "propose":
        raise WorkerProtocolError(
            "unsupported-operation",
            "the language worker supports only health and propose",
        )
    text = request.get("text")
    if not isinstance(text, str):
        raise WorkerProtocolError("invalid-request", "text must be a string")
    if len(text) > MAX_REQUEST_CHARS:
        raise WorkerProtocolError(
            "request-too-large", f"text exceeds {MAX_REQUEST_CHARS} characters"
        )
    brands = _string_list(request.get("brands"), field="brands")
    addresses = _string_list(request.get("addresses"), field="addresses")
    max_length = request.get("max_length")
    proposal = proposer.propose(
        text,
        brands=brands,
        addresses=addresses,
        max_length=None if max_length is None else int(max_length),
    )
    return {"id": request_id, "ok": True, "proposal": proposal.to_dict()}


def _error_response(request_id: Any, exc: Exception) -> dict[str, Any]:
    code = exc.code if isinstance(exc, WorkerProtocolError) else "worker-error"
    return {
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": str(exc)},
    }


def run_jsonl(input_stream: TextIO, output_stream: TextIO, proposer: LocalSpellModel) -> int:
    """Serve JSONL until EOF; malformed lines do not terminate the worker."""

    for raw_line in input_stream:
        # Windows PowerShell may prepend a UTF-8 BOM when piping the first
        # object.  Treat it as transport metadata, never as part of JSON/text.
        line = raw_line.lstrip("\ufeff")
        if not line.strip():
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise WorkerProtocolError("invalid-request", "request must be a JSON object")
            request_id = request.get("id")
            response = process_request(request, proposer)
        except (json.JSONDecodeError, WorkerProtocolError, TypeError, ValueError) as exc:
            response = _error_response(request_id, exc)
        except Exception as exc:  # keep the long-lived worker available to later requests
            response = _error_response(request_id, exc)
        output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
        output_stream.write("\n")
        output_stream.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V7 offline Vietnamese proposal worker")
    parser.add_argument("--model-dir", help="Pinned local Hugging Face snapshot directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-length", type=int, default=256, help="16..512 tokens")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        proposer = LocalSpellModel(
            args.model_dir,
            device=args.device,
            max_length=args.max_length,
        )
    except Exception as exc:
        sys.stdout.write(
            json.dumps(_error_response(None, exc), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        sys.stdout.flush()
        return 2
    return run_jsonl(sys.stdin, sys.stdout, proposer)


if __name__ == "__main__":
    raise SystemExit(main())
