import sys
import os
# Add project root to sys.path to allow direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import argparse
from src.Transformer_pytorch import config
from src.Transformer_pytorch.dataset import JazzDataset
from src.Transformer_pytorch.model import JazzTransformer
from src.Transformer_pytorch.generator import JazzGenerator

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

def save_to_midi(melody, filename, bpm=120):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Settings
    mid.ticks_per_beat = 480
    # Grid: 48 per bar (4/4) -> 12 per beat.
    # 1 grid unit = 480 / 12 = 40 ticks
    ticks_per_grid = 40
    
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    
    # Collect all events (Note On / Note Off) with absolute grid time
    midi_events = []
    for pitch, dur, start in melody:
        if pitch >= 128: continue # Skip rests
        
        # Note On
        midi_events.append({
            'time': start * ticks_per_grid,
            'type': 'note_on',
            'note': pitch,
            'velocity': 100
        })
        
        # Note Off
        midi_events.append({
            'time': (start + dur) * ticks_per_grid,
            'type': 'note_off',
            'note': pitch,
            'velocity': 0
        })
        
    # Sort by time
    midi_events.sort(key=lambda x: (x['time'], 0 if x['type']=='note_off' else 1))
    
    # Convert to Delta Time
    last_time = 0
    for event in midi_events:
        delta = event['time'] - last_time
        track.append(Message(event['type'], note=event['note'], velocity=event['velocity'], time=delta))
        last_time = event['time']
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    mid.save(filename)
    print(f"Saved MIDI to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate Jazz Solos with Transformer')
    parser.add_argument('--checkpoint', type=str, default='models/transformer/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--output', type=str, default='output/midi/transformer_solo.mid', help='Output MIDI filename')
    parser.add_argument('--bpm', type=int, default=120, help='BPM for MIDI file')
    args = parser.parse_args()

    # 1. Load Dataset (needed for Vocab & Duration mappings)
    print("Loading Dataset metadata...")
    try:
        dataset = JazzDataset()
    except FileNotFoundError:
        print("Data not found. Please run preprocessing first.")
        return

    # 2. Initialize Model
    print("Initializing Model...")
    model = JazzTransformer(num_dur_classes=dataset.vocab_size_duration)
    
    # 3. Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")

    # 4. Initialize Generator
    generator = JazzGenerator(model, dataset)
    
    # 5. Define Chord Progression (F Bebop Blues, 12 bars)
    progression = [
        (5, 2, 48),  # F7
        (10, 2, 48), # Bb7
        (5, 2, 48),  # F7
        (0, 1, 24),  # C-7
        (5, 2, 24),  # F7
        (10, 2, 48), # Bb7
        (10, 2, 48), # Bb7
        (5, 2, 48),  # F7
        (8, 1, 24),  # A-7
        (2, 2, 24),  # D7
        (7, 1, 48),  # G-7
        (0, 2, 48),  # C7
        (5, 2, 24),  # F7
        (2, 2, 24),  # D7
        (7, 1, 24),  # G-7
        (0, 2, 24),  # C7
    ]
    
    print(f"\nGenerating solo for F Blues (temp={args.temp})...")
    melody = generator.generate(progression, temperature=args.temp)
    
    print(f"Generated {len(melody)} events")
    save_to_midi(melody, args.output, args.bpm)

if __name__ == "__main__":
    main()
