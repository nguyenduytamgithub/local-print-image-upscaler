# V5 primary sources, revisions and format policy

V5 downloads model data during setup and does not commit checkpoints to Git.
Normal rendering resolves only the pinned local cache and does not contact a
model service. Code and model terms remain those of their authors.

## Segmentation and recognition

- Meta SAM 2 official repository and checkpoints (Apache-2.0):
  <https://github.com/facebookresearch/sam2>
- SAM 2 research page: <https://ai.meta.com/research/sam2/>
- Hugging Face Transformers SAM 2 documentation:
  <https://huggingface.co/docs/transformers/en/model_doc/sam2>
- V5 SAM snapshot: `facebook/sam2.1-hiera-base-plus` at commit
  `b7320756a13354e7530a63935656d35b2f91a290`.
- Grounding DINO official repository (Apache-2.0):
  <https://github.com/IDEA-Research/GroundingDINO>
- V5 Grounding DINO snapshot: `IDEA-Research/grounding-dino-tiny` at commit
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`.
- Tesseract OCR official repository (Apache-2.0):
  <https://github.com/tesseract-ocr/tesseract>
- Official Tesseract installation guidance for Windows, which points to the
  UB Mannheim installer used by the pinned WinGet package:
  <https://tesseract-ocr.github.io/tessdoc/Installation.html>
- Official `tessdata_best` Vietnamese LSTM model (Apache-2.0), pinned to
  revision `e2aad9b983032bb1beff9133104a67cdbb87ca4d` with SHA-256
  `B6B49293D95D0B6DBD8780174627E82C75BE957B6F4ED9862155540D6B00BB45`:
  <https://github.com/tesseract-ocr/tessdata_best>

All Transformers models are loaded from revision-pinned `safetensors` without
`trust_remote_code`. Grounding DINO labels are only semantic hints; SAM masks
and deterministic rules decide pixel boundaries. `setup_v5.ps1` downloads and
verifies `vie.traineddata`; if Tesseract is absent, it installs the exact
`UB-Mannheim.TesseractOCR` WinGet package version recorded in
`MODEL_SHA256SUMS.txt`. `-SkipTesseract` is the deliberate restricted-machine
fallback; V5 then records OCR as unavailable and continues without claiming
text recognition.

## Background reconstruction

- LaMa official repository and checkpoints (Apache-2.0):
  <https://github.com/advimman/lama>
- LaMa WACV 2022 paper:
  <https://openaccess.thecvf.com/content/WACV2022/papers/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.pdf>
- Source of the verified TorchScript conversion and reference preprocessing
  logic (Apache-2.0):
  <https://github.com/enesmsahin/simple-lama-inpainting>
- Adapter TorchScript conversion URL:
  <https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt>
- Expected SHA-256:
  `7BA7AA7AC37A4D41FDBBEBA3A2AF7EAD18058552997E3A3CD1A3B2210C9E6B4C`.

The adapter model is a third-party TorchScript conversion, not a file hosted by
the original LaMa repository. V5 loads that verified file directly and does
not install the adapter package, whose historical Pillow/NumPy dependency
metadata conflicts with the pinned V5 runtime. The fixed hash is a supply-chain
guard, not a new license or proof of model equivalence. Poster mode prefers
deterministic colour/gradient/structure reconstruction; LaMa is reserved for
texture.

## Editable formats

- OpenRaster baseline layer-stack specification:
  <https://www.openraster.org/baseline/layer-stack-spec.html>
- OpenRaster baseline file layout:
  <https://www.openraster.org/baseline/file-layout-spec.html>
- Adobe Photoshop format overview and PSD/PSB limits:
  <https://helpx.adobe.com/photoshop/desktop/save-and-export/export-files-to-different-formats/photoshop-file-formats-overview.html>
- `psd-tools` documentation/repository (MIT):
  <https://psd-tools.readthedocs.io/en/latest/>
  and <https://github.com/psd-tools/psd-tools>
- Canva design-import API:
  <https://www.canva.dev/docs/connect/api-reference/design-imports/>
- Canva Photoshop design-import announcement:
  <https://www.canva.com/newsroom/news/highly-requested-launches/>
- Canva Magic Layers product announcement, used only as a commercial reference
  point for flat-image-to-editable-layer workflows:
  <https://www.canva.com/newsroom/news/magic-layers/>

OpenRaster plus the PNG/mask/manifest ZIP is the open canonical representation.
V5 writes the OpenRaster 0.0.6 baseline layout and validates its stack order,
hierarchy, offsets and flattened composite. PSD is a compatibility adapter and
is limited to 30,000 px per axis plus a conservative 1.6 GB projected raw-layer
budget; V5 does not emit a fake PSB when those limits are exceeded. Canva and
Photoshop import behavior can change and must be tested with the generated
bundle; V5 does not claim that raster OCR layers become native Canva text.
Canva Magic Layers is proprietary product behavior, not an implementation or
model source for V5. V5 remains local/offline after setup and does not use Canva
code, models or services.

Neither OpenRaster nor PSD can restore information that was already flattened
away. Layer grouping and the pixels synthesized beneath removed objects are
recorded as inference in the manifest, never as the recovered original design.
There is no interchange standard that reconstructs a discarded layer graph;
OpenRaster 0.0.6 standardizes packaging of the inferred result only.
