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

## Runtime text topology and matte refinement

- OpenCV contour hierarchy documentation (`RETR_CCOMP`/`RETR_TREE`), used to
  preserve and audit nested foreground/background regions:
  <https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html>
- ViTMatte official repository and reference implementation (MIT):
  <https://github.com/hustvl/ViTMatte>
- Hugging Face Transformers ViTMatte documentation, including the required
  image-plus-trimap input contract:
  <https://huggingface.co/docs/transformers/model_doc/vitmatte>
- Hugging Face model card/distribution metadata for the selected Composition-1k
  small checkpoint (Apache-2.0 metadata):
  <https://huggingface.co/hustvl/vitmatte-small-composition-1k>
- V5 ViTMatte snapshot: `hustvl/vitmatte-small-composition-1k` at commit
  `53222614392e8bd24ed804fbd2f9a43c46ac3850`, with `model.safetensors`
  SHA-256
  `BDA9289DB1BB6762D978B42D1C62AE3F34DAF7497171A347A1D09657EFD788CB`.

OpenCV and the pinned ViTMatte-S checkpoint are **runtime dependencies**, not
research citations alone. The deterministic OpenCV/NumPy pass first estimates
local panel colour, removes colour-supported O/0 counters, G/C apertures and
N-like negative space, and preserves immutable source-ink cores. Compact
Vietnamese marks omitted by a proposal can be restored only through spatial
alignment to a glyph anchor plus Lab-colour agreement; long rules and remote
fragments are rejected. Contour hierarchy and connected-component counts are
recorded for topology QA.

ViTMatte-S then receives a narrow trimap around that fixed mask through the
pinned Transformers runtime. It predicts fractional alpha inside the accepted
topology only: sure foreground and negative space remain hard constraints, and
the result is clipped before export. CUDA is selected when PyTorch reports it
available; the same revision also has a slower CPU path. Setup downloads the
revision-pinned snapshot once, validates the safetensors hash, and normal runs
resolve it from the local cache without network access.

The final deterministic pass is performed on the exact source-resolution layer
stack, not on the model output in isolation. OpenCV connected components find
only area-one alpha islands promoted by the recomposition solver; local CIE Lab
background evidence and a reliable glyph-body palette decide whether a pixel
may be transferred to the parent layer. Parent/background targets are rebuilt,
the flattened stack must remain exact within one 8-bit level, and the automatic
transfer budget is capped at two pixels per text layer.

After that detector converges, V5 freezes the serialized x1 alpha and plans the
actual PSD/ORA z-order backwards. For opaque lower colour `B`, foreground `F`
and alpha `A`, it uses Pillow's integer result
`floor((A*F + (255-A)*B + 127) / 255)`. The preferred clean plate is projected
to the nearest per-channel colour for which an exact integer foreground exists;
alpha is never enlarged to force a colour match. Runtime gates then require
byte-identical canonical alpha plus spatial foreground-component and enclosed
background-hole correspondence at levels 64, 128 and 192. This solver is local
deterministic NumPy/Pillow code; it does not add a cloud service or undeclared
model dependency.

## Mask refinement references and evaluated alternatives

- HQ-SAM official repository/paper implementation:
  <https://github.com/SysCV/sam-hq>
- SAMRefiner official repository:
  <https://github.com/linyq2117/SAMRefiner>
- Matting Anything official repository (CVPR 2023):
  <https://github.com/SHI-Labs/Matting-Anything>
- LayerD official repository (ICCV 2025):
  <https://github.com/CyberAgentAILab/LayerD>
- Guided image filtering paper (ECCV 2010):
  <https://mmlab.ie.cuhk.edu.hk/2010/eccv10_Guided.pdf>

The projects and paper in this section are evaluated alternatives or design
references, not undeclared runtime dependencies. OpenCV and ViTMatte-S are the
explicit exceptions documented in the runtime section above. V5's release path
keeps the pinned SAM 2.1 proposals because, on the bundled poster regression
image, their exact recorded components already isolate the icons cleanly. A
pinned local LayerD evaluation was also performed: it produced only three broad
layers on that artwork and lost/damaged fine poster text, so it was not
substituted blindly for the working multi-object grouping. The delivery pass
instead requires exact component provenance, colour/topology-clean text,
anchor-and-colour protection for Vietnamese marks, topology-constrained
fractional alpha and a hard alpha-ownership gate.

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
