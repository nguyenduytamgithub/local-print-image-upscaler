# Primary sources and model policy

- Swin2SR official repo: https://github.com/mv-lab/swin2sr (Apache-2.0)
- Swin2SR official x4 release: https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1
- HAT official repo: https://github.com/XPixelGroup/HAT (Apache-2.0)
- HAT official model folder: https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0
- Real-ESRGAN official repo: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause)
- Real-ESRGAN x4plus official release: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0
- Spandrel model loader: https://github.com/chaiNNer-org/spandrel (MIT plus permissive architecture licenses)
- PyTorch official install matrix: https://pytorch.org/get-started/previous-versions/
- Perception-distortion tradeoff: https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html

Default policy: fidelity first for text and prices. A perceptual/GAN model may be
selected only after full-image and worst-crop QA. Diffusion restoration is not a
default because it can invent detail and several leading checkpoints have
non-commercial or model-dependent licenses.

The final master does not use a diffusion model. Text, prices and Vietnamese
diacritics must remain exact; a diffusion restorer can generate plausible but
incorrect strokes. The three retained checkpoints are listed with exact hashes
in `MODEL_SHA256SUMS.txt`, and their license notices are retained in
`THIRD_PARTY_LICENSES`.
