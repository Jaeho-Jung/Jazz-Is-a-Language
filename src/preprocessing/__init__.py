"""
WJD Preprocessing Package

Modular preprocessing pipeline for Weimar Jazz Database (WJD) bebop/hardbop solos.
"""

from .config import *
from .chord_parser import ChordParser
from .rhythm_processor import RhythmProcessor
from .data_loader import WJDDatabase
from .feature_engineer import FeatureExtractor
from .augmentation import DataAugmentor

__version__ = "1.0.0"
__all__ = [
    'ChordParser',
    'RhythmProcessor',
    'WJDDatabase',
    'FeatureExtractor',
    'DataAugmentor',
]
