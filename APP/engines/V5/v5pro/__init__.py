"""Production-oriented V5 layer reconstruction.

The original V5 implementation is intentionally retained as a legacy
fallback.  This package implements the exhaustive proposal ledger, atomic
pixel ownership and fail-closed quality gates used by the production path.
"""

from .schema import AlphaCrop, DocumentGraph, ElementNode, ProposalRecord

__all__ = ["AlphaCrop", "DocumentGraph", "ElementNode", "ProposalRecord"]
