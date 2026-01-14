"""
Utility Functions for WJD Preprocessing

Shared helper functions used across modules.
"""

import pandas as pd
from typing import Optional, Any


def clip_to_range(value: float, min_val: float, max_val: float) -> float:
    """
    Clip a value to a specified range.
    
    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
    
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value or default
    """
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_round(value: float, default: int = 0) -> int:
    """
    Safely round a float to integer.
    
    Args:
        value: Value to round
        default: Default value if value is NaN
    
    Returns:
        Rounded integer or default
    """
    if pd.isna(value):
        return default
    return round(value)
