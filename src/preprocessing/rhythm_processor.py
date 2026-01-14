"""
Rhythm Processing and Grid Quantization

Handles rhythm quantization using WJD's structural rhythm information.
"""

import pandas as pd
from .config import GRID_PER_BEAT, GRID_PER_BAR
from .utils import safe_int, safe_divide, clip_to_range


class RhythmProcessor:
    """Processor for rhythm quantization and grid calculations."""
    
    @staticmethod
    def calculate_position_grid(beat: int, tatum: int, division: int) -> int:
        """
        Calculate position in 48-grid system using WJD's structural rhythm info.
        
        This robust method avoids onset-based calculation errors due to micro-timing.
        
        Formula: Pos = (beat - 1) * 12 + Round((tatum - 1) / division * 12)
        
        Args:
            beat: Beat number in bar (1-4 for 4/4 time)
            tatum: Tatum number within beat (1-indexed)
            division: Division of the beat
        
        Returns:
            Position in 48-grid (0-47)
        """
        beat = safe_int(beat, 1)
        tatum = safe_int(tatum, 1)
        division = safe_int(division, 1)
        
        # Avoid division by zero
        if division == 0:
            division = 1
        
        # Calculate position
        pos = (beat - 1) * GRID_PER_BEAT + round((tatum - 1) / division * GRID_PER_BEAT)
        
        # Clip to valid range
        pos = int(clip_to_range(pos, 0, GRID_PER_BAR - 1))
        
        return pos
    
    @staticmethod
    def calculate_duration_grid(duration_sec: float, beat_dur_sec: float) -> int:
        """
        Convert duration from seconds to 12-grid units.
        
        Args:
            duration_sec: Duration in seconds
            beat_dur_sec: Duration of one beat in seconds
        
        Returns:
            Duration in grid units (clipped to max 48)
        """
        if pd.isna(duration_sec) or pd.isna(beat_dur_sec) or beat_dur_sec == 0:
            return 1  # Default to minimum duration
        
        # Convert to grid units
        dur_grid = round(safe_divide(duration_sec, beat_dur_sec, 1.0) * GRID_PER_BEAT)
        
        # Clip to valid range (minimum 1, maximum 48)
        dur_grid = int(clip_to_range(dur_grid, 1, GRID_PER_BAR))
        
        return dur_grid
    
    @staticmethod
    def validate_rhythm_data(beat: int, tatum: int, division: int) -> bool:
        """
        Validate rhythm data for consistency.
        
        Args:
            beat: Beat number
            tatum: Tatum number
            division: Division value
        
        Returns:
            True if valid, False otherwise
        """
        if pd.isna(beat) or pd.isna(tatum) or pd.isna(division):
            return False
        
        try:
            beat = int(beat)
            tatum = int(tatum)
            division = int(division)
            
            # Basic validation
            if beat < 1 or beat > 4:  # 4/4 time
                return False
            if tatum < 1:
                return False
            if division < 1:
                return False
            if tatum > division:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
