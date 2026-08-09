# Vendored LayerD

- Upstream: <https://github.com/CyberAgentAILab/LayerD>
- Commit: `21aef937a0371614adb4d961f52d02409cb8ecc7`
- Paper: *LayerD: Decomposing Raster Graphic Designs into Layers*, ICCV 2025
- License: Apache-2.0; see `LICENSE` and `NOTICE` in this directory.

Only the upstream Python package, build metadata, license, and notice are
vendored. Training data and checkpoints are not included. V5 downloads the
model from the revision-pinned `cyberagent/layerd-birefnet` repository and
verifies the model weight SHA-256 before use.

Local compatibility changes:

1. `layerd.__init__` exposes only the low-level `LayerD` API, avoiding an eager
   optional Pydantic pipeline import that fails under upstream's declared
   Python 3.10/3.11 support range.
2. `TypedDict`/`NotRequired` use `typing_extensions` for Python 3.11.
3. Transformers 5 receives the empty tied-weight mapping omitted by the
   pinned pre-Transformers-5 BiRefNet class; that model has no tied weights.

The decomposition and matting/inpainting algorithms are otherwise unchanged.
