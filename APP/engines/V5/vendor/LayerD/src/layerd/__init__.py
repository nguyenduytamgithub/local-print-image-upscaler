"""LayerD: Decomposing Raster Graphic Designs into Layers.

Vendored from CyberAgentAILab/LayerD at commit
``21aef937a0371614adb4d961f52d02409cb8ecc7``.

The upstream package imports its optional Pydantic pipeline eagerly.  V5 uses
the low-level decomposition model and its own audited exporter, so this local
compatibility shim keeps the import surface deliberately small.  It also lets
the supported Python 3.11 runtime use LayerD without installing an unrelated
web/API dependency.
"""

from layerd.models.layerd import LayerD

__all__ = ["LayerD"]
