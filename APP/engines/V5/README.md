# V5 Pro — reviewable raster layer reconstruction

V5 Pro reconstructs one flattened bitmap as an inferred hierarchy of real
raster layers. It is designed for local poster/graphic editing, not for
recovering the original PSD. A flat PNG or JPEG contains neither the discarded
layer graph nor pixels hidden behind visible objects; V5 therefore labels the
clean background as synthesized and never claims a perfect-recovery rate.

The active public engine is `layer_engine_v5_pro.py`. It combines an independent
element inventory, semantic proposals, geometric reconstruction, LayerD
proposals, ownership arbitration, local review, clean-plate synthesis and
fail-closed QA. V2, V3, V4 and V7 remain separate.

## Recommended workflow

Run these commands from the `RESIZE` root:

```powershell
# Install once, then verify the pinned local runtime
.\APP\engines\V5\setup_v5.ps1
.\APP\engines\V5\setup_v5.ps1 -CheckOnly

# 1. Build the source-resolution editing master
.\upscale layers poster.png 1 --review gui

# 2. Reopen review later if QA still says REVIEW_REQUIRED
.\upscale review ".\OUTPUT\V5_LAYERS\poster_V5_LAYERS_x1"

# 3. Review saves a checkpoint; rerun the same layers command to rebuild outputs
.\upscale layers poster.png 1 --review gui
```

The `review`/`duyet` command never rewrites the existing PSD or ORA. It writes
each decision to `_KY_THUAT\LAYER_REVIEW.json`. On the next `layers` run, the
launcher resumes only when the original path/hash, scale and complete proposal
ID ledger still match. A checkpoint from another image or a changed inventory
is rejected instead of being applied approximately.

Groups created in the review UI are exported as real pass-through groups in
both PSD and ORA. To preserve the bottom-to-top composite, V5 accepts a review
group only when all selected layers share the same immediate parent and occupy
one contiguous run in sibling order. A cross-parent or non-contiguous selection
is rejected rather than silently reordering artwork; grouping does not flatten
the member layers.

V5 also creates deterministic pass-through organizational folders over broad,
contiguous ownership/source runs. They never merge leaves or change hierarchy,
mask, pixels or bottom-to-top order. The two audited K bundles report: `sau`, 44
automatic folders over 999 leaves, root entries 761 to 7 and total sibling entries
1,018 to 63; `truoc`, 76 folders over 881 leaves, root entries 323 to 7 and total
sibling entries 910 to 105. Both remain `REVIEW_REQUIRED`: these figures measure
document-tree organization only and do not prove clean or semantically correct
layer decomposition. A valid explicit user group has precedence. If a later
hierarchy makes an old user group unsafe (different immediate parents or a
non-contiguous run), V5 dissolves
only that container, records the reason and leaves its atomic members ungrouped;
resume cannot resurrect it.

The review canvas initially stays clean; selecting a list row draws only that
item. Enable **Hiện toàn bộ mục chờ duyệt** for the unresolved overview,
or **Hiện cả mục kỹ thuật/đã xử lý** to inspect the complete ledger.
These are display filters: hidden items remain in the checkpoint, proposal
accounting and QA.

After review and manual editing, flatten the edited PSD/ORA to a PNG and upscale
that finished composition:

```powershell
.\upscale high edited.png 4

# Or create the V4 print-delivery bundle with physical size and printer profile
.\upscale print edited.png 4 --width-mm 3000
```

This `V5 x1 -> review -> edit -> flatten -> upscale` path keeps the editable
document manageable. Run V5 directly at x4 only when the layered canvas itself
must already be large:

```powershell
.\upscale layers poster.png 4 --review gui
```

V5 `layers` accepts integer factors only: `n=1..20`. At `n=1`, V5 does not
require the V3 super-resolution checkpoints. At integer `n=2..20`, the launcher
uses the audited local V3 native-x4 master and resamples once to the requested
delivery size. Creating that V3 master for the first time needs NVIDIA CUDA and
all three local V3 checkpoints. For a fractional target such as x1.5 or x2.5,
build/review/edit V5 at x1, flatten the edited document, then pass that PNG to
`upscale high` or `upscale print`.

## Command surface

```text
.\upscale layers <image-or-directory> <n>
                  [--detail exhaustive|grouped]
                  [--review gui|defer|auto]
                  [--inpaint auto|poster|lama]
                  [--no-semantic] [--allow-huge]

.\upscale review <bundle-or-LAYER_REVIEW.json>
.\upscale duyet  <bundle-or-LAYER_REVIEW.json>
```

- The public V5 scale range is the integer set `n=1..20`; decimal/fractional
  factors are rejected by the `layers` workflow.
- `--detail exhaustive` is the production default. V5 Pro maintains exhaustive
  proposal accounting in both accepted detail modes and does not truncate the
  document to a hard layer count.
- Legacy `--max-layers` input remains parser-compatible but is ignored by the
  Pro resource plan and engine. It must not be used as a completeness claim.
- `--review gui` opens the local interface for one image. Directory jobs convert
  GUI review to `defer` so they do not open many windows.
- `defer` and `auto` write a checkpoint without opening the UI. The legacy
  `strict` spelling remains parser-compatible but currently has no distinct V5
  behaviour and does not force a decision for every node. Use `gui`, `defer` or
  `auto` to describe the supported behaviour; unresolved records still produce
  `REVIEW_REQUIRED`.
- `--inpaint poster` favors deterministic colour/gradient/structure fitting;
  `lama` uses the verified local LaMa checkpoint for texture; `auto` selects a
  path from observed image characteristics.
- `--no-semantic` disables the DINO/SAM/BiRefNet semantic branch. It does not
  turn LayerD or OCR into ground truth.
- `--allow-huge` passes the 120 MP warning only after resource checks; the 300 MP
  output hard cap remains.

For a directory, every image uses the same scale/options and runs sequentially:

```powershell
.\upscale layers "D:\BO ANH" 1 --review defer
```

Review each bundle, then rerun the same directory command. Resume matching is
performed separately for every source image.

## What each backend is allowed to do

1. OpenCV/Tesseract inventory records text, QR, frames, rules and other edge
   islands. OCR supplies geometry/metadata; recognized text remains raster.
   Text extraction compares interior and outside-ring background palettes, then
   applies component, row-band and adjacent-object purity gates. An impure text
   or price mask is unresolved and non-move-safe, never silently auto-confirmed.
2. Grounding DINO proposes semantic boxes. SAM 2.1 supplies box-prompted masks,
   and BiRefNet HR-matting can refine accepted product boundaries. An explicit
   failed refinement may retain a bounded fallback mask for review, but that
   node is always unresolved and non-move-safe; confidence cannot auto-confirm it.
   A clean product union without independent instance evidence is labeled
   `atomicity_unverified`/`whole_visible_union`. `compound_subassembly` is used
   only when independent instance evidence explicitly reports `count > 1`.
   Either union may remain move-safe as the whole visible group, but is unresolved,
   never auto-confirmed/auto-atomic. V5 does not invent hidden boundaries between
   touching/occluded items. The review UI exposes this as a friendly per-node
   explanation rather than requiring users to inspect raw metadata.
3. Deterministic geometry reconstructs a frame only from a dominant interior
   hole plus clean side/bottom evidence, producing canonical straight, closed
   topology. Attached ribbon/text is rejected from the frame authority; an
   uncertain candidate remains unresolved. Lines and panels follow their own
   shape evidence while higher-priority content remains protected.
4. The pinned LayerD source/model generates raw foreground mattes and a candidate
   background. These are proposals only. V5 subtracts higher-priority owners and
   records every component in the ledger. A cluster that fails the coherence/
   auto-safety policy is never exported as one contaminated union: its final
   connected pieces become separate atomic unresolved, non-move-safe nodes. A
   review-group ID records their relationship without merging their pixels.
5. Ownership arbitration ensures one visible pixel does not silently belong to
   multiple exported elements. Ambiguous nodes/proposals remain reviewable.
6. Clean-plate synthesis removes accepted layer footprints, audits ghost/seam
   risk and runs source-resolution OCR on the proposed lower background.
7. Export/QA reopens PSD/ORA, checks the actual 8-bit stack and writes a
   machine-readable status for the launcher's publish guard.

LayerD is an active local runtime dependency, but it is never the sole authority
for final layer count, masks, hierarchy or background. A broad LayerD region is
not proof that the design originally had that layer, and a missing LayerD region
is not proof that a small detail should be discarded. Only a cluster that passes
the recorded policy may be consolidated and auto-confirmed.

## Clean full-canvas background and source remainder

V5 Pro deliberately separates background editability from exact source/master
appearance:

- the bottom background is a synthesized clean plate over the **entire canvas**;
  unowned source pixels are never baked back into that layer merely to make the
  preview match;
- the complement of active categorical `visible_alpha` ownership is copied
  from the source/master into a real review layer named
  `REVIEW - SOURCE REMAINDER (hide to reveal clean background)`. It is stacked
  above extracted nodes, so it can preserve exact source pixels inside a clean
  geometry node's hidden `full_support` without creating a second visible
  owner;
- every accepted panel/frame/line with a fitted reference records a
  `geometry_cleanliness` policy. Its standalone PNG uses that reference across
  the entire `full_support`; all observed RGB deviations and antialias pixels
  that cannot be reproduced exactly by the reference are carved out of
  `visible_alpha` and transferred to `SOURCE REMAINDER`. Hiding the remainder
  therefore reveals clean editable geometry rather than a faint text/price
  ghost. Residual-derived geometry without this evidence is unresolved and
  non-move-safe. QA audits the exported `full_support`, not merely the carved
  visible alpha: a frame needs one coherent ring with a dominant interior hole,
  a panel must remain solid, and a line must not carry substantial structure on
  its cross-axis.

Keep `SOURCE REMAINDER` visible when the original source/master appearance must
be preserved for comparison, scaling or continued editing. Hide it to reveal
the full clean background and clean geometry surfaces. The layer can
intentionally include visually quiet
regions: at xN, two samples from the upscaled master may differ even where the
source looked flat, so preserving the remainder makes recomposition
deterministic for that run.

This is a fidelity/safety layer, **not** a semantic object. It is exported as an
unknown, non-move-safe node and must not be moved as though it were a logo,
product or text layer. It deliberately starts unresolved and therefore makes
the bundle `REVIEW_REQUIRED` while it awaits a user decision. Keeping or
accepting it acknowledges a source-fidelity fallback; it does not prove those
pixels were semantically decomposed or recover the original design graph.

## Fail-closed status

V5 Pro uses the exact status strings below:

- `PASS`: proposal accounting is complete; no node remains unresolved; ownership
  is exclusive; alpha stays inside semantic envelopes; clean-plate, recomposition
  and required container round-trip gates pass.
- `REVIEW_REQUIRED`: no hard failure was detected, but proposals/nodes — including
  an unresolved `SOURCE REMAINDER`, geometry without a clean reference or an
  impure text mask — residual background OCR or clean-plate evidence still need
  a decision. This is a normal review outcome, not a hung render.
- `FAIL`: a hard gate failed, such as overlapping ownership, escaped alpha, empty
  exported nodes, failed clean-plate/container checks, recomposition error over
  the configured one-level 8-bit limit, or an automatic-safety contradiction.
  A node cannot claim `auto_confirmed` or move-safe when its metadata records a
  failed semantic refinement, rejected/high-risk LayerD consolidation, or
  missing/failed geometry-cleanliness or text-purity policy. A semantic compound
  cannot claim atomic/auto-confirmed status, and exported geometry that violates
  its frame/panel/line `full_support` topology is also a hard contradiction.

`REVIEW_REQUIRED` is still atomically published to `OUTPUT\V5_LAYERS` so the
user can inspect the evidence, review the checkpoint and rerun. A hard `FAIL`
is not published and cannot replace the last good output. A close-looking
preview or perfect flattening alone cannot prove that the decomposition is
complete or uncontaminated. For a published bundle, always read
`_KY_THUAT\QA_REPORT.html` and `manifest.json`.

## Output bundle

For `PASS` and `REVIEW_REQUIRED`, the published directory
`OUTPUT\V5_LAYERS\<name>_V5_LAYERS_xN\` contains:

```text
00_HUONG_DAN_MO_FILE.txt
01_<name>_EDITABLE.psd          optional compatibility adapter
02_<name>_MASTER.ora            OpenRaster 0.0.6 editable master
03_<name>_PREVIEW.png           flattened QA reference
04_<name>_CLEAN_BACKGROUND.png  synthesized clean plate over the full canvas
05_<name>_LAYERS.zip            portable layer/mask/inventory/QA package
06_<name>_CONTACT_SHEET.png     single sheet or bounded multipage cover
CONTACT_SHEETS\page_*.png       present only for paginated large jobs
LAYERS\                         cropped RGBA assets
MASKS\                          matching alpha masks
ELEMENT_INVENTORY.csv
ELEMENT_INVENTORY.json
manifest.json
_KY_THUAT\                      review checkpoint and QA evidence
```

Small jobs retain the single contact-sheet file. Large jobs paginate at no more
than 32 cards per page; the cover and pages are each bounded to 4,096 pixels per
axis and 16 megapixels. Page assets are recorded by the outer bundle manifest.
The portable `05_*_LAYERS.zip` contract is unchanged and does not absorb the
paginated contact-sheet directory.

The PSD/ORA stack and `LAYERS\` assets include the top `SOURCE REMAINDER` when
source pixels remain outside categorical visible ownership, including observed
deviations carved out of clean geometry. The default preview keeps that layer
visible for source/master fidelity; `04_*_CLEAN_BACKGROUND.png` shows the
underlying clean plate that is revealed when the review layer is hidden.

The PSD adapter is omitted above 30,000 px on either axis or when projected raw
layer data exceeds 1.6 GB. V5 never writes PSD bytes under a fake `.psb`
extension. ORA plus cropped PNG/mask assets are the open interchange path when
PSD is unsuitable. Canva/Photoshop import behaviour can change; test the actual
bundle rather than assuming compatibility from a suffix.

The preview is not an editable master and its presence does not mean `PASS`.
V5 documents are finite RGB raster intermediates, not CMYK/PDF/X print files and
not resolution-independent vectors.

## Truth and editing limits

- V5 creates real pixel layers, not renamed files, but their grouping is inferred.
- A close or exact preview can be produced by `SOURCE REMAINDER` while semantic
  decomposition is still incomplete; preview fidelity is not semantic proof.
- OCR labels do not reconstruct the original font or create native Canva text.
- Hidden pixels are synthesized from visible context. They cannot be recovered
  exactly because they are absent from the flattened source.
- OpenRaster standardizes exchange of the inferred stack; it does not validate
  the inference or recreate a discarded layer graph.
- `PASS` is a statement about the implemented gates, not a 99.99% guarantee,
  infinite zoom or recovery of the original design file.
- Prices, phone numbers, QR codes, legal text, logos and print-critical edges
  must still be inspected at 100% before delivery.

## Install and offline checks

```powershell
.\APP\engines\V5\setup_v5.ps1
.\APP\engines\V5\setup_v5.ps1 -CheckOnly

# Deliberate CPU-only environment
.\APP\engines\V5\setup_v5.ps1 -CpuOnly

# Restricted machine: omit Tesseract deliberately and keep the flag for checks
.\APP\engines\V5\setup_v5.ps1 -SkipTesseract
.\APP\engines\V5\setup_v5.ps1 -CheckOnly -SkipTesseract
```

The supported setup envelope is Windows 10/11 x64 with 64-bit CPython 3.11-3.13.
Setup uses CUDA 12.6 wheels when `nvidia-smi` is available and otherwise creates
an isolated CPU environment. V5 x1 can run on CPU, but DINO, SAM, BiRefNet and
LayerD are substantially slower.

`model_setup.py` downloads and verifies five pinned Hugging Face weights:

- `ZhengPeng7/BiRefNet_HR-matting`;
- `cyberagent/layerd-birefnet`;
- `facebook/sam2.1-hiera-base-plus`;
- `IDEA-Research/grounding-dino-tiny`;
- `hustvl/vitmatte-small-composition-1k`.

The first four participate in the current Pro/semantic paths. The ViTMatte
snapshot remains setup-verified for compatibility with the earlier V5 matte
path; the active `layer_engine_v5_pro.py` does not use it as decomposition
authority. Setup also verifies LaMa, the Vietnamese `vie.traineddata`, and the
pinned Tesseract Windows installer when installation is needed.

Normal rendering resolves model snapshots from the local cache and does not
upload user artwork. `-CheckOnly` is offline and verifies dependency pins,
imports, revisions and recorded SHA-256 values. See `SOURCES.md` and
`MODEL_SHA256SUMS.txt` for the exact provenance recorded by the current source.
