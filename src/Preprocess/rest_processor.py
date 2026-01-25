"""
Rest Processor for WJD Preprocessing

Handles rest event detection, compression, and timing calculation.
Uses absolute position (bar * GRID_PER_BAR + pos_grid) for accurate duration calculation.
"""

import pandas as pd
from typing import Tuple

from .config import GRID_PER_BAR, REST_CHORD_REL_PITCH


class RestProcessor:
    """
    Handles rest event detection and compression.
    
    Key concepts:
    - pos_grid: Relative position within bar (0-47 for 4/4 time)
    - absolute_pos: bar * 48 + pos_grid (used for duration calculation)
    - Rest compression: Consecutive rest beats → single rest event
    """
    
    def process_rests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for rest processing.
        
        Flow:
        1. Calculate absolute position for all events
        2. Identify rest regions (consecutive NaN pitch beats)
        3. Compress consecutive rests into single events
        4. Calculate pos_grid and dur_grid for rest events
        5. Return merged dataframe with notes and compressed rests
        
        Args:
            df: DataFrame with columns including bar, pos_grid, dur_grid, pitch
        
        Returns:
            DataFrame with rest events added (no NaN pos_grid/dur_grid)
        """
        df = df.copy()
        
        # 1. Calculate absolute position for all events
        df['absolute_pos'] = df['bar'] * GRID_PER_BAR + df['pos_grid']
        
        # 2. Mark rest regions
        df['is_rest'] = df['pitch'].isna()
        
        # 3. Identify consecutive rest groups
        df['rest_group'] = (df['is_rest'] != df['is_rest'].shift(1)).cumsum()
        
        # 4. Separate notes and rests
        note_df = df[~df['is_rest']].copy()
        rest_beats_df = df[df['is_rest']].copy()
        
        if len(rest_beats_df) == 0:
            # No rests, just return notes
            df = df.drop(columns=['absolute_pos', 'rest_group'])
            return df
        
        # 5. Compress consecutive rests and calculate timing
        compressed_rests = self._compress_and_calculate(note_df, rest_beats_df)
        
        # 6. Merge notes and compressed rests
        result = pd.concat([note_df, compressed_rests], ignore_index=True)
        
        # 7. Sort by absolute position to maintain order
        result = result.sort_values('absolute_pos').reset_index(drop=True)
        
        # 8. Clean up temporary columns
        result = result.drop(columns=['absolute_pos', 'rest_group'], errors='ignore')
        
        return result
    
    def _compress_and_calculate(self, note_df: pd.DataFrame, rest_beats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compress consecutive rest beats, calculate timing, and split into beat segments.
        
        For each rest group:
        - Keep only the first beat's metadata
        - Calculate pos_grid from previous note's end
        - Calculate dur_grid from next note's start
        - Split long rests into 12-tick segments
        
        Args:
            note_df: DataFrame of note events
            rest_beats_df: DataFrame of rest beats (to be compressed)
        
        Returns:
            DataFrame of rest events with calculated timing
        """
        compressed_rests = []
        
        # Get unique rest groups
        rest_groups = rest_beats_df['rest_group'].unique()
        
        for group_id in rest_groups:
            group = rest_beats_df[rest_beats_df['rest_group'] == group_id]
            first_rest_beat = group.iloc[0]
            
            # Find surrounding notes
            prev_note, next_note = self._find_surrounding_notes(
                note_df, first_rest_beat['absolute_pos']
            )
            
            # Calculate rest timing using absolute positions
            rest_pos_grid, rest_dur_grid, rest_bar = self._calculate_rest_timing(
                prev_note, next_note, first_rest_beat
            )
            
            # Split long rests into beat-aligned segments
            segments = self._split_single_rest(rest_pos_grid, rest_dur_grid, rest_bar)
            
            # Create rest events for each segment
            for seg in segments:
                rest_event = first_rest_beat.copy()
                rest_event['bar'] = seg['bar']
                rest_event['pos_grid'] = seg['pos_grid']
                rest_event['dur_grid'] = seg['dur_grid']
                rest_event['absolute_pos'] = seg['bar'] * GRID_PER_BAR + seg['pos_grid']
                rest_event['is_rest'] = True
                rest_event['pitch'] = pd.NA
                compressed_rests.append(rest_event)
        
        if not compressed_rests:
            return pd.DataFrame()
        
        return pd.DataFrame(compressed_rests)
    
    def _split_single_rest(self, pos_grid: int, dur_grid: int, bar: int) -> list:
        """
        Split a single rest into beat-aligned 12-tick segments.
        
        Algorithm:
        1. Fill to next beat boundary (pos % 12 == 0)
        2. Add full 12-tick beats
        3. Add remaining partial beat
        
        Examples:
        - Rest(pos=6, dur=30) → [(6,6), (12,12), (24,12)]
        - Rest(pos=42, dur=48) → [(42,6,bar), (0,12,bar+1), (12,12), (24,12), (36,6)]
        
        Args:
            pos_grid: Starting position in bar (0-47)
            dur_grid: Total duration in grids
            bar: Starting bar number
        
        Returns:
            List of dicts with pos_grid, dur_grid, bar
        """
        BEAT_GRIDS = 12  # One beat = 12 grids
        
        segments = []
        remaining = dur_grid
        current_pos = pos_grid
        current_bar = bar
        
        while remaining > 0:
            # Calculate distance to next beat boundary
            if current_pos % BEAT_GRIDS == 0:
                # Already at beat boundary, use full beat
                dist_to_boundary = BEAT_GRIDS
            else:
                # Distance to next beat (12, 24, 36, 48)
                next_beat = ((current_pos // BEAT_GRIDS) + 1) * BEAT_GRIDS
                dist_to_boundary = next_beat - current_pos
            
            # Segment duration: min of distance to boundary, remaining, or 12
            segment_dur = min(dist_to_boundary, remaining)
            
            segments.append({
                'pos_grid': current_pos,
                'dur_grid': segment_dur,
                'bar': current_bar
            })
            
            remaining -= segment_dur
            current_pos += segment_dur
            
            # Handle bar boundary (pos >= 48)
            if current_pos >= GRID_PER_BAR:
                current_pos = current_pos % GRID_PER_BAR
                current_bar += 1
        
        return segments
    
    def _find_surrounding_notes(self, note_df: pd.DataFrame, rest_absolute_pos: float) -> Tuple[pd.Series, pd.Series]:
        """
        Find the previous and next note events surrounding a rest.
        
        Args:
            note_df: DataFrame of note events
            rest_absolute_pos: Absolute position of the rest
        
        Returns:
            Tuple of (previous_note, next_note) as Series
            Either can be None if at start/end of solo
        """
        # Previous note: last note before rest
        prev_notes = note_df[note_df['absolute_pos'] < rest_absolute_pos]
        prev_note = prev_notes.iloc[-1] if len(prev_notes) > 0 else None
        
        # Next note: first note after rest
        next_notes = note_df[note_df['absolute_pos'] > rest_absolute_pos]
        next_note = next_notes.iloc[0] if len(next_notes) > 0 else None
        
        return prev_note, next_note
    
    def _calculate_rest_timing(self, prev_note: pd.Series, next_note: pd.Series, first_rest_beat: pd.Series) -> Tuple[int, int, int]:
        """
        Calculate pos_grid and dur_grid for a rest event.
        
        Formula:
        - rest_start_absolute = prev_note_absolute + prev_note_dur_grid
        - rest_bar = rest_start_absolute // GRID_PER_BAR
        - rest_pos_grid = rest_start_absolute % GRID_PER_BAR
        - rest_dur_grid = next_note_absolute - rest_start_absolute
        
        Args:
            prev_note: Previous note event (or None if at start)
            next_note: Next note event (or None if at end)
            first_rest_beat: First beat of the rest region
        
        Returns:
            Tuple of (pos_grid, dur_grid, bar)
        """
        # Calculate rest start position (absolute)
        if prev_note is not None:
            prev_absolute = prev_note['absolute_pos']
            prev_dur = prev_note['dur_grid'] if pd.notna(prev_note['dur_grid']) else GRID_PER_BAR
            rest_start_absolute = prev_absolute + prev_dur
        else:
            # Rest at beginning of solo - use first rest beat position
            rest_start_absolute = first_rest_beat['bar'] * GRID_PER_BAR
        
        # Calculate rest end position (absolute)
        if next_note is not None:
            rest_end_absolute = next_note['absolute_pos']
        else:
            # Rest at end of solo - use last rest beat position + one beat
            last_beat_absolute = first_rest_beat['bar'] * GRID_PER_BAR + (first_rest_beat['beat'] - 1) * (GRID_PER_BAR // 4)
            rest_end_absolute = last_beat_absolute + (GRID_PER_BAR // 4)  # One beat duration
        
        # Convert absolute position back to bar and pos_grid
        rest_bar = int(rest_start_absolute // GRID_PER_BAR)
        rest_pos_grid = int(rest_start_absolute % GRID_PER_BAR)
        
        # Calculate duration
        rest_dur_grid = int(rest_end_absolute - rest_start_absolute)
        
        # Ensure non-negative duration
        if rest_dur_grid <= 0:
            rest_dur_grid = GRID_PER_BAR // 4  # Default to one beat
        
        return rest_pos_grid, rest_dur_grid, rest_bar
