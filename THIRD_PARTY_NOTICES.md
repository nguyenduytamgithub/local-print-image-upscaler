# Third-party notices

Local Print Image Upscaler orchestrates independent open-source projects and
pretrained checkpoints. Their copyrights and licenses remain with their
respective authors.

| Component | Project | License used by the code/project |
|---|---|---|
| Real-ESRGAN | <https://github.com/xinntao/Real-ESRGAN> | BSD-3-Clause |
| Real-ESRGAN NCNN Vulkan | <https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan> | MIT |
| Swin2SR | <https://github.com/mv-lab/swin2sr> | Apache-2.0 |
| HAT | <https://github.com/XPixelGroup/HAT> | Apache-2.0 |
| Spandrel | <https://github.com/chaiNNer-org/spandrel> | MIT and architecture-specific notices |
| G'MIC | <https://gmic.eu/> | CeCILL / CeCILL-C; see the upstream package and <https://cecill.info/> |
| PyTorch | <https://github.com/pytorch/pytorch> | BSD-style license; installed from upstream wheels |
| uv | <https://github.com/astral-sh/uv> | MIT or Apache-2.0 |

Full license texts retained for the local engines are under:

- `APP/engines/V2/THIRD_PARTY_LICENSES/`
- `APP/engines/V3/THIRD_PARTY_LICENSES/`

Pretrained checkpoint terms can differ from the repository code license.
Public packaging should download checkpoints from their official release pages,
record SHA-256 hashes, and avoid re-hosting a checkpoint unless redistribution
permission has been verified.

The public source repository does not redistribute G'MIC, PyTorch, model
checkpoints or other third-party binaries. A future installer must fetch them
from their official sources and retain every notice required by the downloaded
package.
