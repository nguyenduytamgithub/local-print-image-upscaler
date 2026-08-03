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
| uv | <https://github.com/astral-sh/uv> | MIT OR Apache-2.0 |
| SAM 2.1 | <https://github.com/facebookresearch/sam2> | Apache-2.0 |
| Grounding DINO | <https://github.com/IDEA-Research/GroundingDINO> | Apache-2.0 |
| LaMa | <https://github.com/advimman/lama> | Apache-2.0 |
| simple-lama-inpainting adapter | <https://github.com/enesmsahin/simple-lama-inpainting> | Apache-2.0 |
| Tesseract OCR | <https://github.com/tesseract-ocr/tesseract> | Apache-2.0 |
| UB Mannheim Windows build of Tesseract | <https://tesseract-ocr.github.io/tessdoc/Installation.html> | Tesseract distribution; component notices remain applicable |
| Tesseract `tessdata_best` Vietnamese model | <https://github.com/tesseract-ocr/tessdata_best> | Apache-2.0 |
| Transformers | <https://github.com/huggingface/transformers> | Apache-2.0 |
| psd-tools | <https://github.com/psd-tools/psd-tools> | MIT |
| OpenRaster | <https://www.openraster.org/> | Open specification; implementations retain their own licenses |

Retained license texts and source/license records for the local engines are under:

- `APP/engines/V2/THIRD_PARTY_LICENSES/`
- `APP/engines/V3/THIRD_PARTY_LICENSES/`
- `APP/engines/V5/SOURCES.md` records pinned SAM 2.1/Grounding DINO revisions, LaMa and
  Vietnamese OCR hashes, plus the PSD/OpenRaster format policy.

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

V4 Deep Print reuses the local V3 model set and runs the HAT sharper checkpoint in its
second restoration pass. The Apache-2.0 license listed for the HAT source repository does
not by itself replace any separate notice or redistribution term attached to a pretrained
checkpoint; release packaging must verify both.

The source repository does not redistribute G'MIC, PyTorch, model checkpoints, resvg,
Scribus, or other third-party binaries. An installer must fetch them from official sources,
verify expected versions/hashes and retain all required notices.
