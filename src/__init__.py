"""HAM10000 skin-lesion classification — reusable source modules.

Sub-modules
-----------
config         : paths, constants, and the ``DX_FULL`` mapping.
data           : metadata loading, image-index construction, dataset audit.
eda            : exploratory-data-analysis plotting helpers.
stats          : hypothesis tests (Chi-square, Kruskal–Wallis, Dunn post-hoc).
features       : 50 handcrafted image descriptors.
preprocessing  : encoding, stratified split, scaling, SMOTE.
models         : baseline model factory + unified evaluator.
tuning         : LightGBM randomized-search with checkpointing (SMOTE-in-CV).
melanoma       : melanoma-focused improvements (binary gate, two-tier, thresholds).
plotting       : confusion matrices, ROC curves, feature importance.

The modules intentionally expose *small* focused functions so that the
notebooks under ``Report/``, ``Milestone/``, and ``Proposal/`` stay readable
and so that reviewers can re-run any stage in isolation.
"""

from . import config  # noqa: F401

__all__ = ["config"]
__version__ = "1.0.0"
