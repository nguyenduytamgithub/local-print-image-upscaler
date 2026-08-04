# V7 Design Repair

V7 is for a flat poster/catalogue whose text is broken, blurry or misspelled.
It is not another global sharpening preset.  It detects text, requests approval
when meaning is uncertain, removes the approved old glyphs, reconstructs only
their declared footprint, and renders clean Unicode text at the final size.

## Safety contract

- PP-OCRv6 and Tesseract are evidence, not truth.
- Repeated PP-OCRv6 augmentations are one engine, never multiple independent
  votes. Tesseract only votes when the verified Vietnamese tessdata is present;
  V7 never labels an English fallback as Vietnamese evidence.
- Prices, phone numbers, addresses, SKUs, digits and brand names are never
  silently corrected.
- Any language-model change remains a proposal until approved.
- Background pixels outside the edit footprint must remain byte-identical at
  source scale.
- An unresolved or failed-QA bundle is marked as such and is not called a
  print-ready final.
- SVG text is genuinely editable.  Raster pictures remain raster; V7 never
  renames a PNG to pretend that it is vector artwork.
- A loose OCR rectangle is not trusted as typography geometry.  V7 measures the
  bounded old-ink mask, shapes candidates with RAQM/HarfBuzz, searches supported
  OpenType variable `wght`/`wdth` axes and hard-rejects missing Vietnamese glyphs.
- Geometry, centre, clipping, ghost and reconstruction seams are release gates.
  OCR readback remains an advisory because OCR can misread correctly rendered
  accents at very small raster sizes.
- A review file is bound to the exact source SHA-256 and to a per-region digest
  of source-space bbox plus NFC OCR text. Missing or stale authority fails closed.
- The source-gated font face, variable-font axes and size ratio are locked for
  x2..x20. Final geometry and lock fidelity must pass before status can be PASS.

## Runtime split

`V7/.venv` contains the official Windows CPU Paddle runtime.  This avoids DLL,
NumPy and oneDNN conflicts with V3/V5.  The RTX GPU remains available to the
separate V3 PyTorch process for restoration/upscaling and the optional local
Vietnamese suggestion model.  Model inference is sequential so a 12 GB RTX
3060 is sufficient.

Run setup once from PowerShell:

```powershell
cd C:\Users\Admin\Desktop\RESIZE
powershell -ExecutionPolicy Bypass -File .\APP\engines\V7\setup_v7.ps1
```

Then use the public command:

```powershell
.\upscale repair poster.png 4
.\upscale repair poster.png 4 --review defer
.\upscale repair poster.png 4 --review-file "C:\path\TEXT_REVIEW.json"
.\upscale repair poster.png 4 --review strict --review-file "C:\path\TEXT_REVIEW.json"
```

For a folder, V7 never opens dozens of dialogs. It writes one review JSON per
image. Approve/edit those files, then rerun the same folder command: the launcher
snapshots and reuses each matching review automatically. `strict` requires an
explicit decision for every region; `defer` may auto-render only independently
green regions.
