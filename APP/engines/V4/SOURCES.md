# V4 implementation sources

V4 `print` is a disclosed mixed vector/raster pipeline as supported by SVG and
PDF/X-4; it is not a renamed PNG container. Its visible print layer is produced
locally from a validated V3 native x4 master followed by a second, uniform HAT
inference pass. Overlapping tiles are used only to control VRAM; neural x4
predictions are antialiased and blended directly into the requested xN canvas,
without allocating or writing a full x16 intermediate.

The G'MIC/VTracer path reconstruction is retained as a separate editable layer at
opacity `0` by default. This is intentional: automatic color tracing represents
continuous gradients and photographic texture as discrete stacked regions and can
introduce posterization, banding or contour noise. V4 `vector` is the raster-free
mode for flat artwork. Every manifest identifies and hashes the source, V3 master,
Deep raster, model and outputs, and records image/path counts and QA metrics.

- HAT / Activating More Pixels in Image Super-Resolution Transformer, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html
- Official HAT implementation and pretrained model documentation, Apache-2.0:
  https://github.com/XPixelGroup/HAT
- VTracer 0.6.15, MIT: https://github.com/visioncortex/vtracer
- G'MIC 4.0.2, CECILL v2.1: https://gmic.eu/
- resvg 0.47.0, MIT OR Apache-2.0: https://github.com/linebender/resvg
- Scribus 1.6.6, GPL-2.0-or-later: https://www.scribus.net/
- pikepdf 10.11.0, MPL-2.0: https://github.com/pikepdf/pikepdf
- OpenCV, Apache-2.0: https://opencv.org/
- ISO 15930-7 PDF/X-4 overview: https://www.iso.org/standard/55843.html
- PDF Association technical requirements for PDF/X:
  https://pdfa.org/technical-side-and-requirements-of-pdfx/
- Adobe print workflow guidance for PDF/X-4:
  https://helpx.adobe.com/nz/illustrator/using/creating-pdf-files.html
- SVG 2 embedded content (`image`) and vector structure:
  https://www.w3.org/TR/SVG/embedded.html
- GWG Sign & Display: https://gwg.org/sign-display/

ISO 15930-7 defines PDF/X-4 as a PDF 1.6 complete-exchange format. The local
pikepdf checks confirm selected structural properties only; they are not a full
conformance certification. A production file still needs the print provider's
Acrobat/callas/Enfocus or GWG preflight profile.

The default `ISO Coated v2 300% (basICColor)` OutputIntent is a generic
coated-paper reference. It must not be treated as a device/media profile for
PVC, tarpaulin or another wide-format substrate. The printer's supplied ICC
profile or PDF preset remains authoritative.

The executable runtimes are machine-local and excluded from Git. Versions and
hashes are recorded in each V4 manifest.

Pinned Windows installer/archive SHA-256 values used by `setup_v4.ps1`:

- G'MIC 4.0.2 CLI ZIP: `6D0F553C383A93CED4B006B067A49277F45AD69A030287ADDA7ABD4D4DE15163`
- resvg 0.47.0 win64 ZIP: `5684E59CEAA53CE720B49EFB441B0918AE99D04E8CE3F6F753664524592D67F1`
- Scribus 1.6.6 win64 installer: `4C7313DA22B8DAA025DAB0A2D57E82E8B1827C8A46A02DB1D4E78B6037B40264`
