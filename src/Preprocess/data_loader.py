"""
Database Loading and Data Extraction

Handles SQLite database operations, data loading, and chord alignment.
"""

import sqlite3
import pandas as pd
from typing import List, Dict
from .config import TARGET_MELODY_TYPE, TARGET_STYLES, EXCLUDED_STYLES, EXCLUDED_MELIDS, TARGET_SIGNATURE, NOTE_NAMES

class WJD:
    """Manager class for WJDDatabase"""

    def __init__(self, db_path:str):
        """
        Initialize database manager.

        Args:
            db_path (str): Path to the database file
        """
        self.db_path = db_path

    def get_target_melids(self) -> List[int]:
        """
        Extract melids that satisfies the following conditions:
        - melody_type.type: {TARGET_MELODY_TYPE}
        - solo_info.style: {TARGET_STYLES}
        - beats.signature: {TARGET_SIGNATURE}

        Returns:
            List[int]: List of melids
        """
        conn = sqlite3.connect('data/wjazzd.db')
        
        # Build query dynamically
        placeholders_styles = ','.join(['?'] * len(TARGET_STYLES))
        placeholders_excluded_styles = ','.join(['?'] * len(EXCLUDED_STYLES))
        placeholders_excluded_melids = ','.join(['?'] * len(EXCLUDED_MELIDS))
        
        query = f"""
                SELECT DISTINCT m.melid
                FROM melody_type m
                JOIN solo_info s ON m.melid = s.melid
                JOIN beats b ON m.melid = b.melid
                WHERE m.type = ?
                AND s.style IN ({placeholders_styles})
                AND s.style NOT IN ({placeholders_excluded_styles})
                AND m.melid NOT IN ({placeholders_excluded_melids})
                AND b.signature = ?
                ORDER BY m.melid
        """
        
        params = [TARGET_MELODY_TYPE] + \
                 TARGET_STYLES + \
                 EXCLUDED_STYLES + \
                 EXCLUDED_MELIDS + \
                 [TARGET_SIGNATURE]
                 
        df = pd.read_sql_query(
            query, 
            conn,
            params=params
        )
        conn.close()

        return df['melid'].tolist()

    def load_solo_data(self, melid: int) -> Dict[str, pd.DataFrame]:
        """
        Load data for a single solo

        Args:
            melid (int): ID of the solo

        Returns:
            pd.DataFrame: Dataframe containing the solo data
        """
        conn = sqlite3.connect(self.db_path)

        # Load melody events
        melody_query = """
        SELECT eventid, melid, pitch, onset, duration, bar, beat, tatum, division, beatdur
        FROM melody
        WHERE melid = ?
        ORDER by eventid
        """
        melody_df = pd.read_sql_query(melody_query, conn, params=(melid,))

        # Load beats events
        beats_query = """
        SELECT beatid, melid, bar, beat, chord, signature, chorus_id 
        FROM beats
        WHERE melid = ?
        ORDER by bar, beat
        """
        beats_df = pd.read_sql_query(beats_query, conn, params=(melid,))

        # Load solo_info events
        solo_info_query = """
        SELECT melid, key, avgtempo, performer, style, title 
        FROM solo_info
        WHERE melid = ?
        """
        solo_info_df = pd.read_sql_query(solo_info_query, conn, params=(melid,))

        conn.close()

        return {
            'melody': melody_df,
            'beats': beats_df,
            'solo_info': solo_info_df
        }
