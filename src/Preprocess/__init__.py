"""
WJD Preprocessing Package

Modular preprocessing pipeline for Weimar Jazz Database (WJD) bebop/hardbop solos.
"""

from .config import *
from .chord_parser import ChordParser
from .rest_processor import RestProcessor
from .utils import *
from .preprocess import Preprocessor

__version__ = "1.1.0"
__all__ = [
    'ChordParser',
    'RestProcessor',
    'Preprocessor',
    'safe_int',
    'safe_divide',
    'clip_to_range'
]   