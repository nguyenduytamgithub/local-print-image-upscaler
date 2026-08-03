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

Full license texts retained for the local engines are under:

- `APP/engines/V2/THIRD_PARTY_LICENSES/`
- `APP/engines/V3/THIRD_PARTY_LICENSES/`

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
