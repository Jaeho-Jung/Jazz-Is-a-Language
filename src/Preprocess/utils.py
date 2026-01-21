"""
Utility Functions for WJD Preprocessing

Helper functions for data validation, type conversion, and value manipulation.
"""

import pandas as pd
from typing import Union, Any


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int, return default if conversion fails.
    
    Args:
        value: Value to convert
        default: Default value to return if conversion fails
    
    Returns:
        int: Converted value or default
    
    Examples:
        >>> safe_int(5.7)
        5
        >>> safe_int(None, default=0)
        0
        >>> safe_int("invalid", default=-1)
        -1
    """
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, return default if denominator is 0.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division fails
    
    Returns:
        float: Division result or default
    
    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default=0.0)
        0.0
        >>> safe_divide(None, 5, default=1.0)
        1.0
    """
    if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
        return default
    return numerator / denominator


def clip_to_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """
    Clip value to specified range [min_val, max_val].
    
    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clipped value
    
    Examples:
        >>> clip_to_range(5, 0, 10)
        5
        >>> clip_to_range(-5, 0, 10)
        0
        >>> clip_to_range(15, 0, 10)
        10
    """
    return max(min_val, min(max_val, value))
