import os

# Disable download counts for transformers to accelerate
os.environ["HF_UPDATE_DOWNLOAD_COUNTS"] = "FALSE"

from .evaluator import Evaluator
from .utils import parse_argument

__all__ = ["Evaluator", "parse_argument"]
