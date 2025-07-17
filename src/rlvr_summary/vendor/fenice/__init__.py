"""
FENICE (Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction)

This is a vendored copy of FENICE from https://github.com/Babelscape/FENICE
Modified to work with newer transformers versions and integrated directly into the codebase.
"""

from .FENICE import FENICE

__all__ = ["FENICE"]