# V7 technical sources

V7 separates deterministic print reconstruction from text recognition and
language suggestions.  A model is never treated as an authority to alter the
customer's wording.

## Primary specifications and projects

- PaddleOCR 3.x OCR pipeline and PP-OCRv6 model family:
  <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>
- Official PaddleOCR repository (Apache-2.0):
  <https://github.com/PaddlePaddle/PaddleOCR>
- Tesseract 5 documentation and official Vietnamese `tessdata_best`
  (Apache-2.0): <https://tesseract-ocr.github.io/tessdoc/>
- Unicode Standard Annex #15.  V7 uses NFC and deliberately does not apply
  compatibility-folding NFKC to customer text:
  <https://unicode.org/reports/tr15/>
- HarfBuzz shaping model and OpenType features:
  <https://harfbuzz.github.io/shaping-concepts.html>
- OpenType 1.9.1 and OS/2 font-embedding permissions:
  <https://learn.microsoft.com/en-us/typography/opentype/spec/>
- OpenType Font Variations (`fvar`) used for bounded weight/width matching:
  <https://learn.microsoft.com/en-us/typography/opentype/spec/fvar>
- fontTools `ttLib` for cmap, names, metrics and OS/2 inspection:
  <https://fonttools.readthedocs.io/en/latest/ttLib/>

## Research informing the architecture

- Shimoda et al., “De-Rendering Stylized Texts”, ICCV 2021.  Its key lesson
  for V7 is to recover explicit text/style/background parameters and then
  render, rather than repeatedly sharpening damaged raster glyphs:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Shimoda_De-Rendering_Stylized_Texts_ICCV_2021_paper.html>
- FASTER, WACV 2025, a font-agnostic scene-text editing/rendering framework:
  <https://openaccess.thecvf.com/content/WACV2025/html/Das_FASTER_A_Font-Agnostic_Scene_Text_Editing_and_Rendering_Framework_WACV_2025_paper.html>

Generative scene-text projects are research references, not the core content
authority.  They can hallucinate spelling or graphics.  V7 therefore uses
deterministic masking/background fitting/font rendering and exposes every
language-model change for approval.

## Pinned model provenance

- `PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec`: official Paddle inference
  archives from `paddle-model-ecology.bj.bcebos.com`; extracted runtime files
  are verified by `MODEL_SHA256SUMS.txt`.
- `nrl-ai/vn-spell-correction-base`, Apache-2.0, exact Hugging Face revision
  `61596a71696ba360ae828f9db3806610afedf6d3`.  It supplies proposals only.
- Tesseract Vietnamese `tessdata_best` is downloaded into V7's own ignored model
  directory at pinned revision `e2aad9b983032bb1beff9133104a67cdbb87ca4d`
  and verified by SHA-256 before use as an independent vote.
