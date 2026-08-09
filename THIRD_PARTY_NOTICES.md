# Third-party notices

Local Print Image Upscaler orchestrates independent open-source projects and pretrained
checkpoints. Their copyrights and licenses remain with their respective authors.

| Component | Project | Upstream license |
|---|---|---|
| Real-ESRGAN | <https://github.com/xinntao/Real-ESRGAN> | BSD-3-Clause |
| Real-ESRGAN NCNN Vulkan | <https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan> | MIT |
| Swin2SR | <https://github.com/mv-lab/swin2sr> | Apache-2.0 |
| HAT | <https://github.com/XPixelGroup/HAT> | Apache-2.0 |
| Spandrel | <https://github.com/chaiNNer-org/spandrel> | MIT and architecture-specific notices |
| PyTorch | <https://github.com/pytorch/pytorch> | BSD-style |
| G'MIC | <https://gmic.eu/> | CeCILL / CeCILL-C; see the upstream package |
| VTracer | <https://github.com/visioncortex/vtracer> | MIT |
| resvg | <https://github.com/linebender/resvg> | MIT OR Apache-2.0 |
| Scribus | <https://www.scribus.net/> | GPL-2.0-or-later |
| pikepdf | <https://github.com/pikepdf/pikepdf> | MPL-2.0 |
| OpenCV | <https://opencv.org/> | Apache-2.0 |
| Pillow | <https://python-pillow.org/> | HPND |
| NumPy | <https://numpy.org/> | BSD-3-Clause |
| PaddlePaddle | <https://github.com/PaddlePaddle/Paddle> | Apache-2.0 |
| PaddleOCR / PP-OCRv6 source | <https://github.com/PaddlePaddle/PaddleOCR> | Apache-2.0; model-asset terms/notices also apply |
| PaddleX | <https://github.com/PaddlePaddle/PaddleX> | Apache-2.0 |
| fontTools | <https://github.com/fonttools/fonttools> | MIT |
| freetype-py | <https://github.com/rougier/freetype-py> | BSD-3-Clause |
| HarfBuzz | <https://github.com/harfbuzz/harfbuzz> | Old MIT |
| uharfbuzz | <https://github.com/harfbuzz/uharfbuzz> | Apache-2.0; bundled HarfBuzz notices remain applicable |
| regex | <https://github.com/mrabarnett/mrab-regex> | Apache-2.0 additions plus inherited CNRI/Python terms |
| SciPy | <https://github.com/scipy/scipy> | BSD-3-Clause; binary-wheel subcomponent notices apply |
| scikit-image | <https://github.com/scikit-image/scikit-image> | BSD-3-Clause with file-specific BSD-2-Clause/MIT notices |
| uv | <https://github.com/astral-sh/uv> | MIT OR Apache-2.0 |
| SAM 2.1 | <https://github.com/facebookresearch/sam2> | Apache-2.0 |
| Grounding DINO | <https://github.com/IDEA-Research/GroundingDINO> | Apache-2.0 |
| ViTMatte | <https://github.com/hustvl/ViTMatte> | MIT; checkpoint/model-card terms also apply |
| LayerD source | <https://github.com/CyberAgentAILab/LayerD> | Apache-2.0; vendored `LICENSE` and `NOTICE` retained |
| cr-renderer bundled by LayerD | <https://github.com/CyberAgentAILab/cr-renderer> | Apache-2.0 per retained LayerD notice |
| LaMa | <https://github.com/advimman/lama> | Apache-2.0 |
| simple-lama-inpainting adapter | <https://github.com/enesmsahin/simple-lama-inpainting> | Apache-2.0 |
| Tesseract OCR | <https://github.com/tesseract-ocr/tesseract> | Apache-2.0 |
| UB Mannheim Windows build of Tesseract | <https://tesseract-ocr.github.io/tessdoc/Installation.html> | Tesseract distribution; component notices remain applicable |
| Tesseract `tessdata_best` Vietnamese model | <https://github.com/tesseract-ocr/tessdata_best> | Apache-2.0 |
| Transformers | <https://github.com/huggingface/transformers> | Apache-2.0 |
| timm | <https://github.com/huggingface/pytorch-image-models> | Apache-2.0 |
| Kornia | <https://github.com/kornia/kornia> | Apache-2.0 |
| Kornia-rs | <https://github.com/kornia/kornia-rs> | Apache-2.0 |
| BiRefNet HR/LayerD model artifacts | <https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting>, <https://huggingface.co/cyberagent/layerd-birefnet> | Upstream model-card/artifact terms; no license inferred from a source repository |
| `nrl-ai/vn-spell-correction-base` | <https://huggingface.co/nrl-ai/vn-spell-correction-base> | Apache-2.0 as declared by the upstream model card |
| psd-tools | <https://github.com/psd-tools/psd-tools> | MIT |
| OpenRaster | <https://www.openraster.org/> | Open specification; implementations retain their own licenses |

Retained license texts and source/license records for the local engines are under:

- `APP/engines/V2/THIRD_PARTY_LICENSES/`
- `APP/engines/V3/THIRD_PARTY_LICENSES/`
- `APP/engines/V5/SOURCES.md` records pinned SAM 2.1/Grounding DINO/BiRefNet/LayerD revisions,
  ViTMatte compatibility, LaMa and Vietnamese OCR hashes, plus review and PSD/OpenRaster policy.
- `APP/engines/V5/vendor/LayerD/LICENSE`, `NOTICE` and `VENDOR.md` retain LayerD attribution,
  upstream commit `21aef937a0371614adb4d961f52d02409cb8ecc7` and local compatibility notes.
- `APP/engines/V7/SOURCES.md` records the PP-OCRv6 sources, Unicode/OpenType references and
  the exact Vietnamese proposal-model revision; `MODEL_SHA256SUMS.txt` records downloaded assets.

The complete Apache-2.0 text used by the Apache-licensed V5 sources is retained at
`APP/engines/V3/THIRD_PARTY_LICENSES/HAT_APACHE-2.0.txt`; V5 source-specific attribution is
recorded in `APP/engines/V5/SOURCES.md`.

The V4 executable runtimes are machine-local and excluded from Git. Their pinned versions,
official project links and runtime hashes are recorded in `APP/engines/V4/SOURCES.md` and in
each generated manifest. If a release redistributes any V4 binary, that release must also
include the exact license and notices required by the corresponding upstream package.

Pretrained checkpoint terms can differ from the repository code license. Public packaging
should download checkpoints from official release pages, record SHA-256 hashes, and avoid
re-hosting a checkpoint unless redistribution permission has been verified.

V5 Pro vendors the Apache-2.0 LayerD source and downloads the separate
`cyberagent/layerd-birefnet` model at revision
`679f743cd001fb5d6360e59e8e1904678c5fa734`. The source license is not treated as proof that
every model/training asset has identical redistribution terms. LayerD is used only as a proposal
backend; this usage policy does not change its license or the terms of its model dependencies.
The independently downloaded `ZhengPeng7/BiRefNet_HR-matting` snapshot is likewise pinned by
revision/hash, while its model-card/artifact terms remain upstream.

V4 Deep Print reuses the local V3 model set and runs the HAT sharper checkpoint in its
second restoration pass. The Apache-2.0 license listed for the HAT source repository does
not by itself replace any separate notice or redistribution term attached to a pretrained
checkpoint; release packaging must verify both.

V7 downloads official PP-OCRv6 inference assets and a revision-pinned Vietnamese spelling
proposal model during local setup. Repository licenses do not automatically settle every
checkpoint or dataset redistribution question. Public releases must verify the terms attached
to the exact downloaded artifacts and retain the package/model notices. The language model is
used only for review proposals; this usage policy does not alter its upstream license.

The V7 virtual environment is machine-local and excluded from Git. Installed wheels retain
their own `.dist-info` license files, including notices for binary dependencies bundled by
SciPy, scikit-image, Pillow/FreeType/HarfBuzz and Paddle packages. If a release redistributes
that environment or any wheel, it must include all corresponding license files rather than
relying on this summary table alone.

The source repository does not redistribute G'MIC, PyTorch, model checkpoints, resvg,
Scribus, Paddle runtimes or other third-party binaries. An installer must fetch them from
official sources, verify expected versions/hashes and retain all required notices.
