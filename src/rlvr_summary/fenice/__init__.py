"""
FENICE (Factuality Evaluation of Summarization based on Natural Language Inference and Claim Extraction)

Integrated FENICE implementation for factual consistency evaluation.
Originally from https://github.com/Babelscape/FENICE
Modified and integrated into rlvr-summary for research purposes.
"""

from .FENICE import FENICE

__all__ = ["FENICE"]
