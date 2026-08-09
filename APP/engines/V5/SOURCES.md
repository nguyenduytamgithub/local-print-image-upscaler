# V5 Pro sources, pinned revisions and format policy

V5 downloads model data during setup and does not commit checkpoints to Git.
Normal rendering resolves the pinned local cache and does not contact a model
service. Repository-code licenses do not automatically settle every model or
dataset term; those terms remain with each upstream author.

The values below are copied from the current `model_setup.py`,
`MODEL_SHA256SUMS.txt`, `setup_v5.ps1` and runtime source. A SHA-256 check proves
file identity only; it is not a quality score, license grant or proof that an
inferred layer is correct.

## Runtime model inventory

| Role | Repository | Immutable revision | `model.safetensors` SHA-256 |
|---|---|---|---|
| Product-boundary refinement | `ZhengPeng7/BiRefNet_HR-matting` | `5d6b6f8adcb5b417c871b1d84ceaae9871355b7f` | `A5A4DE698739EA5E0E8BBAB28E1B293DDE95092B87A442D566CBC585C53CEF55` |
| LayerD matting backend | `cyberagent/layerd-birefnet` | `679f743cd001fb5d6360e59e8e1904678c5fa734` | `28F8ACBA2736067BF2EB8152F2D3CE75A388DD4E4E57F068B30760A8FE1C44D0` |
| Box-prompted segmentation | `facebook/sam2.1-hiera-base-plus` | `b7320756a13354e7530a63935656d35b2f91a290` | `2012733A0DE5D03EFD1BBA550A2847C4551BE9EF2E0D497C83074DF66189F780` |
| Semantic boxes | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` | `1A2412EF99BD74BCD3C2A246FA1E48581F8889A1300C9051974741314FC042F3` |
| Earlier V5 matte-path compatibility | `hustvl/vitmatte-small-composition-1k` | `53222614392e8bd24ed804fbd2f9a43c46ac3850` | `BDA9289DB1BB6762D978B42D1C62AE3F34DAF7497171A347A1D09657EFD788CB` |

The current `layer_engine_v5_pro.py` path uses the first four entries through
its semantic and LayerD backends. Setup still downloads/verifies the ViTMatte-S
snapshot for compatibility with the earlier V5 matte implementation; the Pro
engine does not treat ViTMatte as decomposition authority.

SAM 2 and Grounding DINO are loaded from revision-pinned local files without
remote code. The BiRefNet HR snapshot includes revision-pinned Transformers
architecture `.py` files and is loaded locally with `trust_remote_code=True`;
setup restricts the downloaded file patterns and the runtime verifies the
recorded safetensors hash before inference.

Primary upstream references:

- Meta SAM 2 official repository (Apache-2.0):
  <https://github.com/facebookresearch/sam2>
- SAM 2 research page: <https://ai.meta.com/research/sam2/>
- Transformers SAM 2 documentation:
  <https://huggingface.co/docs/transformers/en/model_doc/sam2>
- Grounding DINO official repository (Apache-2.0):
  <https://github.com/IDEA-Research/GroundingDINO>
- BiRefNet HR-matting distribution used by the runtime:
  <https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting>
- ViTMatte official source (MIT) and Transformers interface:
  <https://github.com/hustvl/ViTMatte> and
  <https://huggingface.co/docs/transformers/model_doc/vitmatte>

These models can support a clean **visible union** without proving how many
independent source instances created it. Without independent instance evidence,
V5 records `atomicity_unverified` with extent `whole_visible_union`.
`compound_subassembly` is reserved for independent evidence that explicitly
reports `count > 1`. Either union may be move-safe as one visible group, but it
remains unresolved and cannot be auto-confirmed or claimed as atomic. No cited
model authorizes inventing boundaries hidden by touching or occluded products.

## LayerD is a proposal backend, not final authority

- Official LayerD repository and ICCV 2025 implementation:
  <https://github.com/CyberAgentAILab/LayerD>
- Vendored upstream source commit:
  `21aef937a0371614adb4d961f52d02409cb8ecc7`.
- Vendored license: Apache-2.0; the retained `vendor/LayerD/LICENSE`, `NOTICE`
  and `VENDOR.md` record upstream attribution and local compatibility changes.
- Runtime model: `cyberagent/layerd-birefnet` at the revision/hash shown in the
  model table.

V5 Pro calls the low-level pinned LayerD decomposition core locally. LayerD
provides raw foreground mattes and a candidate background. Its output is then
reconciled with OCR/QR/geometry and semantic proposals, subjected to exclusive-
ownership arbitration, exposed to review and checked by clean-plate and container
QA. A raw cluster that fails the recorded coherence/auto-safety policy is split
into separate atomic connected pieces; each remains unresolved and non-move-safe,
and a review-group ID relates them without unioning their pixels. Raw LayerD
layers are not blindly copied to the final document.

This restriction follows the runtime's own recorded limitation: decomposition
of a flat graphic is ill-posed, and tiny text or ambiguous granularity requires
independent inventory and review. A LayerD proposal is evidence, not proof of
the discarded source layer graph.

The vendored LayerD notice also identifies its bundled Apache-2.0 components:
`simple-lama-inpainting` and CyberAgent's `cr-renderer`. The project does not
claim that the separate model artifact inherits a license merely because the
LayerD source repository is Apache-2.0; model-card/artifact terms still apply.
The pinned V5 requirement files also list LayerD/BiRefNet runtime dependencies
`timm==1.0.28`, `kornia==0.8.3` and `kornia-rs==0.1.14`; their installed package
metadata identifies Apache-2.0 and the exact versions remain in
`requirements.lock`.

## OCR, QR and geometry inventory

- Tesseract OCR official repository (Apache-2.0):
  <https://github.com/tesseract-ocr/tesseract>
- Official Windows installation guidance:
  <https://tesseract-ocr.github.io/tessdoc/Installation.html>
- Official `tessdata_best` Vietnamese model (Apache-2.0):
  <https://github.com/tesseract-ocr/tessdata_best>
- OpenCV contour hierarchy reference:
  <https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html>

`vie.traineddata` is pinned to revision
`e2aad9b983032bb1beff9133104a67cdbb87ca4d` with SHA-256
`B6B49293D95D0B6DBD8780174627E82C75BE957B6F4ED9862155540D6B00BB45`.
When installation is required, setup asks WinGet for exact package
`UB-Mannheim.TesseractOCR` version `5.4.0.20240606`; the recorded installer
SHA-256 is
`C885FFF6998E0608BA4BB8AB51436E1C6775C2BAFC2559A19B423E18678B60C9`.
`-SkipTesseract` is an explicit restricted-machine fallback; V5 then records
OCR as unavailable instead of pretending an English fallback is Vietnamese.

OCR strings and semantic labels remain metadata/review hints. They do not create
editable type or authorize changing a price, phone number, legal line or QR.
The release path evaluates text masks against both interior and outside-ring
background palettes, then records component, row-band and adjacent-object
purity evidence. An impure text/price candidate is unresolved and non-move-safe;
OCR confidence alone cannot override that local QA policy.

Frame reconstruction uses the dominant interior hole as topology authority and
requires compatible clean evidence from the side and bottom strokes. The
canonical result is straight and closed; same-colour ribbon/text attached to an
outer contour is not accepted as part of the frame. QA inspects the exported
`full_support`: a frame must have one coherent ring and dominant hole, a panel
must remain solid, and a line must not contain substantial cross-axis structure.
Failure keeps the candidate unresolved or blocks a contradictory automatic claim.

## Background reconstruction

- LaMa official repository and WACV 2022 paper:
  <https://github.com/advimman/lama> and
  <https://openaccess.thecvf.com/content/WACV2022/papers/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.pdf>
- Source of the verified TorchScript conversion and preprocessing logic
  (Apache-2.0): <https://github.com/enesmsahin/simple-lama-inpainting>
- Adapter checkpoint URL:
  <https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt>
- Expected SHA-256:
  `7BA7AA7AC37A4D41FDBBEBA3A2AF7EAD18058552997E3A3CD1A3B2210C9E6B4C`.

The adapter checkpoint is a third-party TorchScript conversion, not a file from
the original LaMa repository. Poster mode prefers deterministic colour,
gradient and structure fitting; LaMa is the verified texture fallback. Neither
path can recover exact pixels that were hidden in the flat source. The clean
plate is therefore marked synthesized and audited for residual/ghost risk.

## Review and fail-closed QA policy

The review server, proposal ledger, ownership arbitration and QA rules are local
project code. The public command is `upscale review <bundle>`; it updates the
checkpoint only. The next `layers` run must verify source identity, scale and
proposal IDs before applying those decisions and rebuilding PSD/ORA.

Review-created groups are materialized as real pass-through groups in both PSD
and ORA. The exporter accepts them only when every member has the same immediate
parent and the members are contiguous in sibling order, so grouping cannot
change the flattened composite.

Automatic organizational folders use the same pass-through constraint over
deterministic contiguous ownership/source runs. They preserve every atomic leaf,
parent, pixel and bottom-to-top order. The audited K artifacts report: `sau`, 44
automatic folders over 999 leaves, root entries 761 to 7 and total sibling entries
1,018 to 63; `truoc`, 76 folders over 881 leaves, root entries 323 to 7 and total
sibling entries 910 to 105. Both remain `REVIEW_REQUIRED`: this is organizational
telemetry, not evidence that the decomposition is clean or semantically correct.
A valid explicit user group has precedence. A stale group that becomes
cross-parent or non-contiguous is dissolved with an
audit record and cannot be resurrected by resume; its leaves are not silently
merged or reassigned to an auto-folder.

`PASS` means the declared accounting, ownership, envelope, clean-plate,
recomposition and container gates passed. It does not prove the original layer
graph, invisible background or a universal accuracy percentage. Diagnostic
bundles with `REVIEW_REQUIRED` are published for human review and are not
approved deliverables. Semantic refinement explicitly rejected by the backend
cannot be auto-confirmed or move-safe. QA treats any automatic-safety claim that
contradicts failed semantic refinement or rejected/high-risk LayerD consolidation
metadata as a hard failure. A hard `FAIL` is rejected by the launcher before
atomic publication and cannot replace the last good output.

Clean geometry uses a validated reference surface over full support. Observed
RGB/antialias deviations are removed from categorical geometry ownership and
preserved in the top `SOURCE REMAINDER` review layer; hiding that layer exposes
the clean surface. Residual-derived panel/frame/line candidates without a clean
reference remain unresolved and non-move-safe. `REVIEW_REQUIRED` is therefore a
normal review outcome, not a claim of complete or 99.99% reconstruction.

## Editable formats

- OpenRaster baseline layer-stack specification:
  <https://www.openraster.org/baseline/layer-stack-spec.html>
- OpenRaster baseline file layout:
  <https://www.openraster.org/baseline/file-layout-spec.html>
- Adobe PSD/PSB format overview:
  <https://helpx.adobe.com/photoshop/desktop/save-and-export/export-files-to-different-formats/photoshop-file-formats-overview.html>
- `psd-tools` documentation/repository (MIT):
  <https://psd-tools.readthedocs.io/en/latest/> and
  <https://github.com/psd-tools/psd-tools>
- Canva design-import API and Photoshop import announcement:
  <https://www.canva.dev/docs/connect/api-reference/design-imports/> and
  <https://www.canva.com/newsroom/news/highly-requested-launches/>
- Canva Magic Layers announcement, used only as a commercial comparison point:
  <https://www.canva.com/newsroom/news/magic-layers/>

V5 writes OpenRaster 0.0.6 and validates hierarchy, offsets and flattened
composite. PSD is a compatibility adapter limited to 30,000 px per axis and a
conservative 1.6 GB projected raw-layer budget; V5 never emits a fake PSB.
Canva/Photoshop import behaviour must be tested with the actual bundle. Canva
Magic Layers is proprietary and V5 uses no Canva code, models or service.

Contact sheets are bounded delivery previews, not decomposition evidence. Small
jobs keep one `06_*_CONTACT_SHEET.png`. Large jobs emit that file as a cover plus
`CONTACT_SHEETS/page_*.png`, at most 32 cards per page and no more than 4,096 px
per axis or 16 MP per image. Page files are included in the outer manifest; the
portable layers ZIP layout remains unchanged.

OpenRaster standardizes exchange of the inferred raster stack. It does not
reconstruct or certify a layer graph that was discarded when the source was
flattened.

## Evaluated references, not undeclared runtime dependencies

- HQ-SAM: <https://github.com/SysCV/sam-hq>
- SAMRefiner: <https://github.com/linyq2117/SAMRefiner>
- Matting Anything: <https://github.com/SHI-Labs/Matting-Anything>
- Guided image filtering paper:
  <https://mmlab.ie.cuhk.edu.hk/2010/eccv10_Guided.pdf>

These entries are design/research references only. They are not silently called
by the V5 Pro release path.
