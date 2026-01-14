"""
WJD Bebop/Hardbop Solo Preprocessing Pipeline

Modular preprocessing pipeline for Weimar Jazz Database (WJD).
Orchestrates specialized modules for data extraction, processing, and augmentation.

Based on requirements:
- requirement-preprocess.md: Functional requirements
- model_data_logic.md: Event-based sequence modeling rationale
- data-format.md: SQLite3 database schema
"""

from pathlib import Path
from typing import List
import pandas as pd

# Import modular components
from src.preprocessing import (
    WJDDatabase,
    FeatureExtractor,
    DataAugmentor
)
from src.preprocessing.config import (
    DB_PATH,
    OUTPUT_PKL,
    OUTPUT_CSV,
    AUGMENTATION_SHIFTS
)


def preprocess_wjd(db_path: str, output_pkl: str, output_csv: str):
    """
    Main preprocessing pipeline orchestrator.
    
    Args:
        db_path: Path to WJD SQLite database
        output_pkl: Output pickle file path
        output_csv: Output CSV file path
    """
    print("=" * 80)
    print("WJD Bebop/Hardbop Solo Preprocessing Pipeline (Modular)")
    print("=" * 80)
    
    # Initialize components
    db = WJDDatabase(db_path)
    feature_extractor = FeatureExtractor()
    augmentor = DataAugmentor()
    
    # Step 1: Extract target melids
    print("\n[Step 1] Extracting target melids...")
    melids = db.get_target_melids()
    
    if len(melids) == 0:
        print("No solos found matching the criteria!")
        return
    
    # Step 2: Process each solo
    print(f"\n[Step 2] Processing {len(melids)} solos...")
    processed_solos = []
    
    for i, melid in enumerate(melids):
        print(f"  Processing melid {melid} ({i+1}/{len(melids)})...")
        
        try:
            # Load solo data
            solo_data = db.load_solo_data(melid)
            
            # Align chords to melody
            solo_data['melody'] = db.align_chords_to_melody(
                solo_data['melody'],
                solo_data['beats']
            )
            
            # Handle pickup measures
            solo_data['melody'] = db.handle_pickup_measures(
                solo_data['melody'],
                solo_data['beats']
            )
            
            # Extract features
            processed_df = feature_extractor.process_solo(solo_data)
            
            # Add original key shift (0)
            processed_df['key_shift'] = 0
            
            processed_solos.append(processed_df)
            
        except Exception as e:
            print(f"    Error processing melid {melid}: {e}")
            continue
    
    print(f"  Successfully processed {len(processed_solos)} solos.")
    
    # Step 3: Data augmentation
    print(f"\n[Step 3] Augmenting data with {len(AUGMENTATION_SHIFTS)} transpositions...")
    augmented_solos = augmentor.augment_dataset(processed_solos, AUGMENTATION_SHIFTS)
    print(f"  Total augmented solos: {len(augmented_solos)}")
    
    # Step 4: Create training dataset
    print("\n[Step 4] Creating training dataset...")
    final_df = feature_extractor.create_training_dataset(augmented_solos)
    
    print(f"  Total training examples: {len(final_df)}")
    print(f"  Features: {list(final_df.columns)}")
    
    # Step 5: Save output
    print("\n[Step 5] Saving output files...")
    
    # Save as pickle
    final_df.to_pickle(output_pkl)
    print(f"  Saved: {output_pkl}")
    
    # Save as CSV
    final_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    
    # Print summary statistics
    print_summary(melids, augmented_solos, final_df)


def print_summary(melids: List[int], augmented_solos: List[pd.DataFrame], final_df: pd.DataFrame):
    """Print preprocessing summary statistics."""
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Original solos: {len(melids)}")
    print(f"Augmentation shifts: {AUGMENTATION_SHIFTS}")
    print(f"Total augmented solos: {len(augmented_solos)}")
    print(f"Total training examples: {len(final_df)}")
    print(f"\nDataset shape: {final_df.shape}")
    print(f"\nSample statistics:")
    print(f"  Pitch range: {final_df['target_pitch'].min()} - {final_df['target_pitch'].max()}")
    print(f"  Duration range: {final_df['target_dur'].min()} - {final_df['target_dur'].max()}")
    print(f"  Position range: {final_df['pos_grid'].min()} - {final_df['pos_grid'].max()}")
    print(f"  Chord qualities: {sorted(final_df['chord_quality'].unique())}")
    print(f"  Styles: {final_df['style'].unique()}")
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"Error: Database not found at {DB_PATH}")
        print("Please ensure the WJD database is in the correct location.")
        exit(1)
    
    # Run preprocessing
    preprocess_wjd(DB_PATH, OUTPUT_PKL, OUTPUT_CSV)
