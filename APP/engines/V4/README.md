# V4 quality-gated print engine (internal)

V4 exposes two deliberately different policies:

1. `print` (default) is **V4 Print**. Its visible print layer is the best
   spatially uniform candidate that passes native-x4 QA and a no-Deep ablation:
   guarded V3 USM, a materially better V3/Deep restoration, or an explicitly
   labelled V3 fallback. A real VTracer path group is retained as an independent
   editing layer, but its opacity is `0` by default.
2. `vector` is a raster-free reconstruction made only from filled Bezier paths.
   It is intended for flat logos and simple artwork, not photographs or gradients.

Neither policy is a filename-extension trick. The SVG explicitly declares its
embedded raster and path groups, the PDF contains the corresponding image/path
objects, and the manifest records hashes, counts and QA results.

## Quality-gated print pipeline

For `upscale print <image> <n>` the launcher:

1. creates or reuses an exact, validated V3 native x4 master;
2. runs `Real_HAT_GAN_sharper` a second time, uniformly over that whole x4 master;
3. tiles only to fit GPU memory, uses overlapping tiles and cosine blending, and
   antialiased-resamples each neural x4 prediction directly into the requested xN
   canvas instead of allocating or saving an unnecessary x16 intermediate;
4. evaluates independent halo-guarded crops at V3-native pixel scale, using the whole
   frame up to the QA memory limit or a deterministic 8x8 sample for larger masters;
   crop pixels are never joined into a metric mosaic or downsampled to source x1;
5. compares a guarded-USM V3 control with optional uniform Deep ablations. A candidate
   must retain structure, bound ringing, improve at least one edge-strength and one
   energy/detail indicator, and pass the complete gate in at least 75% of crops. Deep
   must materially improve crop coverage without worse retention to beat the control;
6. renders the selected xN raster once using finite Gaussian support, halo stripes and
   a memory-mapped output, then re-runs the native-x4 gate on the actual encoded PNG.
   A failed render verification uses and labels the V3 fallback;
7. preprocesses a separate trace surface with G'MIC and reconstructs real paths
   with VTracer, retaining them at opacity `0` for selective editing;
8. renders proofs with resvg, exports through Scribus and publishes the bundle only
   after structural and visual QA passes.

The path group is intentionally non-printing by default. Automatic full-color
tracing discretizes continuous gradients, shadows and photographic texture into
stacked color regions, which can create posterization, banding and contour noise.
Making it visible globally would therefore damage the uniform QA-selected raster. Users
may enable or copy selected paths in a vector editor after visual inspection.

`vector` remains available when raster-free output matters more than photographic
fidelity. It can scale paths without a pixel grid, but cannot infer the original
font or recover continuous tones from a low-resolution bitmap.

## Public command and outputs

```powershell
cd "C:\Users\Admin\Desktop\RESIZE"
.\upscale print poster.png 10 --width-mm 3000
.\upscale vector logo.png 4 --width-mm 3000
```

`print` writes one atomic bundle under `OUTPUT\V4_PRINT`:

```text
<name>_V4_PRINT\
  <name>_EDITABLE.svg       visible QA-selected raster + hidden editable path group
  <name>_PRINT_PDFX4.pdf    print PDF using the selected raster as visible artwork
  <name>_PREVIEW_xN.png     raster proof rendered from the same SVG master
```

Its technical manifest is kept separately under `APP\manifests\V4_PRINT`. The
manifest identifies the source, V3 master, Deep cache, model, GPU, requested scale,
ICC intent, image/path counts, output hashes and QA metrics.

## GPU, runtime and cache

`print` requires an NVIDIA CUDA GPU and the V3 CUDA runtime/model set. On the
project RTX 3060 12 GB, a roughly 1254 x 1254 source rendered at x10 normally takes
about **15–25 minutes on the first run**. This is an observed planning range, not a
guarantee; tile fallback, disk speed, SVG complexity and PDF export can change it.

The validated V3 native x4 and raw Deep xN rasters are cached locally by canonical
sRGB pixel SHA-256, engine/model/config hashes, native-master SHA-256 and requested
scale. Byte-different containers with identical canonical pixels can share AI work,
while each job preserves its current source DPI for print geometry. Repeating the same source at the
same `n` reuses both expensive GPU stages. Editing the source or changing `n`
invalidates the relevant Deep cache. No image or cache is sent to a cloud service.

`vector` does not need CUDA, although tracing complex artwork can be CPU- and
memory-intensive.

## PDF/X, preflight and color

Scribus exports PDF 1.6 with PDF/X-4 identification, page boxes and an embedded
OutputIntent ICC profile. pikepdf then performs a **structural self-check** for the
declared version, OutputIntent, encryption state, page geometry and image/path
objects. That check is not independent PDF/X certification.

Production delivery must still be preflighted with the print shop's own profile,
for example GWG Sign & Display rules in Acrobat Pro, callas pdfToolbox or Enfocus
PitStop. The bundled `ISO Coated v2 300% (basICColor)` output intent is a generic
coated-paper reference, not proof that colors are correct for PVC, tarpaulin, ink,
media and RIP at a particular shop. The shop's ICC profile or supplied PDF preset
is authoritative, followed by a physical color proof.

The exporter selects the smallest integer `1:d` scale for which the longest
finished side plus bleed on both edges is at most 5000 mm on the real PDF page.
It records the final size, bleed and scale; finished-size bleed is divided by the
same denominator. The structural check reopens the PDF and measures MediaBox,
TrimBox and BleedBox against those requested values and the 5000 mm limit.
The Scribus helper also measures the imported SVG group, scales it proportionally
to cover trim plus bleed, centres it, and rejects placement errors above 0.5 mm.

## Tests

Run core tests in the V4 environment, then run Deep raster tests in the V3 CUDA
environment. The required-runtime flag turns a missing Deep runtime into a failure
instead of a successful skip:

```powershell
& .\APP\engines\V4\.venv\Scripts\python.exe -B -m unittest discover `
  -s APP\engines\V4\tests -p "test_v4.py" -v
if ($LASTEXITCODE -ne 0) { throw "V4 core tests failed." }
& .\APP\engines\V4\.venv\Scripts\python.exe -B -m unittest discover `
  -s APP\engines\V4\tests -p "test_cli_safety.py" -v
if ($LASTEXITCODE -ne 0) { throw "V4 CLI safety tests failed." }

$env:RESIZE_V4_REQUIRE_DEEP_RUNTIME = "1"
try {
  & .\APP\engines\V3\.venv\Scripts\python.exe -B -m unittest discover `
    -s APP\engines\V4\tests -p "test_deep_raster_v4.py" -v
  if ($LASTEXITCODE -ne 0) { throw "V4 deep CUDA tests failed or were unavailable." }
}
finally {
  Remove-Item Env:RESIZE_V4_REQUIRE_DEEP_RUNTIME -ErrorAction SilentlyContinue
}
```

## Primary standards and upstream projects

- HAT paper, CVPR 2023: <https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html>
- Official HAT code and pretrained-model documentation: <https://github.com/XPixelGroup/HAT>
- VTracer raster-to-vector project: <https://github.com/visioncortex/vtracer>
- resvg renderer: <https://github.com/linebender/resvg>
- Scribus desktop-publishing/export project: <https://www.scribus.net/>
- ISO 15930-7:2010, PDF/X-4: <https://www.iso.org/standard/55843.html>
- PDF Association, PDF/X technical requirements: <https://pdfa.org/technical-side-and-requirements-of-pdfx/>
- Ghent Workgroup Sign & Display specification: <https://gwg.org/sign-display/>
- W3C SVG embedded-content model: <https://www.w3.org/TR/SVG/embedded.html>

Pinned component versions, archive hashes and licensing notes are in
`SOURCES.md` and the repository `THIRD_PARTY_NOTICES.md`.
