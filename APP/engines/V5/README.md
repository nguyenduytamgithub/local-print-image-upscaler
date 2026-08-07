# V5 Smart Editable Layers

V5 turns one flat bitmap into a deliberately small set of **real raster layers**.
It combines official SAM 2.1 masks, Grounding DINO semantic hints, Tesseract
line geometry, deterministic grouping, pinned ViTMatte-S text-edge refinement
and background reconstruction. It is separate from V2, V3 and V4 and does not
modify their code or outputs.

V5 classifies poster/graphic content separately from photographic or textured
content. Poster masks are grouped into panels, rows, objects and parent-child
stacks instead of hundreds of tiny fragments. Photographic content follows a
more conservative SAM/OCR/DINO selection path. The result is an inferred edit
structure, not the original layer graph that was discarded when the source was
flattened.

## Run

From the `RESIZE` root:

```powershell
# Source-resolution layers: recommended for editing
.\upscale layers poster.png 1

# AI-sharpened x4 layers: requires the existing V3 model set
.\upscale layers poster.png 4

# Process every supported image in a directory
.\upscale layers "D:\BO ANH" 1

# The same scale and options are applied to every image in a batch
.\upscale layers "D:\BO ANH" 4 --max-layers 20
```

Optional expert controls:

```powershell
.\upscale layers poster.png 1 --max-layers 20 --inpaint poster --no-semantic
```

- `--max-layers 4..60`: foreground-layer usability budget; default 24. The
  synthesized background is one additional layer.
- `--inpaint auto|poster|lama`: deterministic colour/gradient fill for artwork,
  or LaMa for photographic texture. `auto` chooses conservatively.
- `--no-semantic`: skip Grounding DINO and group with SAM/OCR geometry only.
- `--allow-huge`: pass the 120 MP warning after checking resources; the 300 MP
  V5 hard cap still applies.

The public scale range is `1 <= n <= 20` and decimal factors are accepted. At
`n=1`, segmentation and editing stay at source resolution. At `n>1`, V5 uses
the audited local V3 native-x4 AI master and resamples once to the requested
final size; all three V3 checkpoint files must therefore be installed.

For a lightweight editing workflow, run V5 at `n=1`, edit the PSD/ORA, export
a flattened PNG from the editor, then run `upscale high <edited.png> <n>` or
`upscale print <edited.png> <n>` for final delivery. Use V5 at `n=4` when the
editable document itself must already have a large pixel canvas.

V5 preserves the requested pixel dimensions, not the source document's
physical size/DPI contract. Its PSD/ORA files are RGB raster editing
intermediates, not CMYK or PDF/X print deliverables. After editing, export a
flattened PNG and use `upscale print ... --width-mm ...` to build the V4 print
bundle, then preflight it against the printer's ICC/profile requirements.

Supported inputs are single-frame PNG, JPEG, WebP, BMP and TIFF. Animated or
multi-page files are rejected. Source alpha is currently composited onto white
before layer inference so the normalized input is unambiguous.

## Model roles

- **SAM 2.1 Base+** proposes pixel regions. It does not know the original PSD
  hierarchy.
- **Grounding DINO Tiny** supplies semantic boxes/labels. Poster geometry still
  determines pixel boundaries; DINO labels are not trusted as masks.
- **Tesseract 5 + pinned `tessdata_best` `vie`** supplies Vietnamese line boxes
  and OCR metadata. Recognition never creates native text objects.
- **Pinned ViTMatte-S Composition-1k** estimates fractional alpha only along
  already-clean raster-text boundaries. Deterministic colour/topology rules,
  not ViTMatte, own counters, apertures, accents and the outer silhouette.
- **LaMa** fills plausible photographic texture only when selected. Poster
  mode instead favors deterministic colour, gradient and structural fill.

The manifest records the exact model revisions, device, selected content path,
grouping policy and background-restoration policy used for each image.
The normalized sRGB profile is embedded byte-for-byte in every RGB/RGBA PNG,
all colour PNG members of the ORA and the PSD document ICC resource. Linear
`L` alpha masks intentionally remain untagged because an RGB profile does not
describe alpha values.

## Clean-matte delivery policy

V5 separates **proposal area** from **editable ownership**. Colour-guided halo
expansion and the wider inpainting work radius may help grouping/restoration,
but neither is allowed to enlarge an exported object alpha. For object-like
layers, V5 resolves the exact recorded `sam_NNNN_cc_NNN` connected components
from the original SAM output and discards unanchored islands. Raster text keeps
separate Vietnamese accents while rejecting long panel/frame rules and isolated
specks. The renderer then clips every non-zero alpha pixel to the final semantic
ownership map; shadows or neighbouring artwork outside it stay on the lower
layer instead of travelling with the object.

Raster text receives a second, deterministic negative-space pass before any
neural matting. V5 estimates the local panel/background colour in CIE Lab from
a ring around the text proposal, preserves background-distant source-ink cores,
and subtracts panel-coloured proposal pixels even when they lie inside one
connected SAM/OCR region. This opens O/0 counters, G/C apertures and the
negative spaces around N-like diagonal strokes instead of blindly filling every
hole. If the colour evidence is weak or the proposed removal is excessive, V5
falls back to the conservative text mask rather than deleting uncertain ink.

Detached Vietnamese marks need the opposite protection. V5 may restore a
compact source-colour component missed by the proposal only when it aligns
above or below a glyph-sized anchor and its Lab colour agrees with that anchor.
Long panel rules, remote specks and colour-mismatched fragments are ineligible;
the manifest records recovered accent components and protected-source-ink
recall.

Only after that topology is fixed does V5 run
`hustvl/vitmatte-small-composition-1k` at revision
`53222614392e8bd24ed804fbd2f9a43c46ac3850`, with verified
`model.safetensors` SHA-256
`BDA9289DB1BB6762D978B42D1C62AE3F34DAF7497171A347A1D09657EFD788CB`.
ViTMatte-S receives a narrow trimap for each text crop. Sure-foreground ink and
all removed negative space are hard constraints, and its result is clipped back
inside the clean source topology. It can estimate fractional pixel coverage;
it cannot refill an O, close a G, invent a remote component or change wording.
CUDA is used when available, including the supported RTX 3060 path; otherwise
the same pinned model runs on CPU more slowly. No artwork or inference request
is sent to a model service.

Before resizing or export, V5 also renders an exact x1 stack and audits the
alpha that the colour-recomposition solver will actually publish. A lone
high-alpha text pixel is reassigned to its parent only when its source matte is
low, local Lab evidence matches the surrounding background, and it does not
match a reliable glyph-body palette. The stack is then rebuilt and checked
again. This closes the gap between a clean model matte and the final PNG/PSD
alpha while protecting punctuation and Vietnamese marks; automatic removal is
hard-capped at two pixels per layer and is recorded in `manifest.json`.

At `n>1`, the fractional source matte is resampled with Lanczos and restricted
to a one-source-pixel antialias envelope around the same topology. This avoids
nearest-neighbour x4 blocks without turning nearby panel lines or shadows into
layer ownership. The exact serialized x1 alpha becomes the canonical scale
source for every layer. V5 then solves the complete PSD/ORA stack backwards in
its real z-order. At each step it moves the preferred inpainted lower colour
only as far as that layer's fixed 8-bit alpha can reproduce the selected master,
then solves an exact 8-bit foreground colour using Pillow's integer compositing
rule. Colour correction is therefore not allowed to raise alpha, close a glyph
counter or create a detached opaque resampling lobe.

`manifest.json` records both `matte_cleanup` and `matte_qa`. Publication stops
if a binary matte escapes semantic ownership, if source ViTMatte alpha escapes
the clean topology, or if a scaled fractional matte escapes its audited narrow
envelope. Every delivered alpha canvas must also equal its canonical scaled
alpha byte-for-byte. Refined layers are compared spatially at alpha 64, 128 and
192 using foreground-8/background-4 connectivity: missing or orphan components,
merged glyphs, closed or invented holes, split counters and intrusion into a
protected hole core all block publication. The source-stack preflight must
converge without exceeding its per-layer safety cap. Flattened fidelity is still
checked independently, so a clean-looking mask cannot pass merely by hiding a
visual mismatch in the preview.

## Output bundle

`OUTPUT\V5_LAYERS\<name>_V5_LAYERS_xN\` contains:

- `*_EDITABLE.psd`: Photoshop/Photopea-oriented pixel layers and groups, also
  suitable for testing with Canva's current PSD importer;
- `*_MASTER.ora`: open OpenRaster 0.0.6 master for Krita/GIMP;
- `*_LAYERS.zip`: portable cropped RGBA PNGs, alpha masks, OCR JSON and a
  portable manifest; it deliberately does not duplicate the PSD/ORA/previews;
- `LAYERS\` and `MASKS\`: directly accessible assets with canvas offsets;
- `*_PREVIEW.png`: recomposed visual result;
- `*_LAYER_MAP.png` and `*_CONTACT_SHEET.png`: grouping QA;
- `TEXT_OCR.json` and `manifest.json`: OCR hints, hashes, models, limits and QA.

The PSD adapter is deliberately disabled above 30,000 px per axis or when the
projected raw layer data exceeds 1.6 GB. `psd-tools` cannot yet be claimed as a
reliable edited PSB writer beyond that boundary; V5 never writes PSD bytes with
a fake `.psb` extension. ORA and the ZIP remain the open canonical outputs.

Open the file that fits the editor:

- Photoshop or Photopea: use the PSD when present;
- Krita or GIMP: use the OpenRaster master;
- Canva: try the PSD importer first. If its current importer does not preserve
  hierarchy/alpha correctly, upload the cropped RGBA files from `LAYERS`
  manually and place them using each `canvas_offset` in `manifest.json`.

The preview is a QA reference, not the editable master.

## What V5 can and cannot recover

- The masks and pixel layers are real; this is not a filename conversion.
- OCR text remains raster pixels. A recognized label is **not** the original
  font and is never presented as editable type.
- Colour/topology cleanup and ViTMatte improve a finite raster boundary; they
  do not rebuild the original font outline, emit SVG paths or provide infinite
  zoom. Truly resolution-independent text/logo output still requires verified
  native type or deliberate vector reconstruction.
- A flat PNG/JPEG does not contain the old layer graph. V5 infers useful groups;
  it cannot prove the original grouping.
- No interchange standard can reconstruct a layer graph that was discarded.
  OpenRaster 0.0.6 standardizes how the inferred layers are exchanged; it does
  not standardize or prove the inference itself.
- Pixels hidden behind an object do not exist in the flat source. V5 fills them
  plausibly and marks the background as synthesized; it cannot recover the
  exact unseen artwork.
- For critical logos, prices, legal text and print jobs, inspect masks and the
  cleaned lower layer at 100% before editing or printing.
- PSD/Canva import behavior belongs to those applications and can change. Test
  the actual generated bundle; a `.psd` suffix alone is not a compatibility
  guarantee.

## Install/check

```powershell
.\APP\engines\V5\setup_v5.ps1
.\APP\engines\V5\setup_v5.ps1 -CheckOnly

# Deliberate CPU-only installation (kept separate from a CUDA V3 runtime)
.\APP\engines\V5\setup_v5.ps1 -CpuOnly

# Deliberately omit OCR on a restricted machine; keep the same flag for checks
.\APP\engines\V5\setup_v5.ps1 -SkipTesseract
.\APP\engines\V5\setup_v5.ps1 -CheckOnly -SkipTesseract
```

The supported setup envelope is Windows 10/11 x64 with 64-bit CPython
3.11-3.13. Setup reuses a compatible CUDA V3 environment when present to avoid
duplicating several gigabytes. On a clean machine, or for a CPU-only install,
it creates the isolated `V5\.venv`. A working `nvidia-smi` selects the official
CUDA 12.6 wheel; otherwise setup selects the official CPU wheel. CPU extraction
works but SAM and ViTMatte-S are much slower.
On a CPU-only machine, use V5 at `n=1`. Creating a new V3 sharpening master
for `n>1` is CUDA-only; an already validated local cache may be reused, but a
clean CPU machine cannot generate that cache.

Setup also checks for Tesseract. When it is missing, setup uses WinGet to
install the revision-pinned Windows package recommended by Tesseract's Windows
documentation, then downloads the exact official `tessdata_best` Vietnamese
model. If Tesseract cannot be installed on a restricted machine, the deliberate
fallback is `setup_v5.ps1 -SkipTesseract`; layer extraction still works, but OCR
grouping/naming hints are disabled. A setup performed with `-SkipTesseract`
must also use that flag with `-CheckOnly`.

`-CheckOnly` is offline: it checks the exact runtime selected by `upscale.cmd`,
all package pins and imports, all three Hugging Face revision/weight hashes, the
LaMa hash, the Vietnamese model hash and an actual Tesseract language-load
command. Model snapshots are pinned to exact revisions and all four large model
artifacts are checked by SHA-256. No user artwork is uploaded.

Use `n=1` on a machine without the V3 super-resolution checkpoint set. `n>1`
uses the audited V3 neural x4 master and therefore needs those three local V3
checkpoints as well.

See `SOURCES.md` for exact upstream projects, revisions, formats and license
notes.
