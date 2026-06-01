import sys
import os
# Add project root to sys.path to allow direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import os
import argparse
from src.RNN_pytorch import config
from src.RNN_pytorch.dataset import JazzDataset
from src.RNN_pytorch.model import JazzRNN
from src.RNN_pytorch.generator import JazzGenerator

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
        
    mid.save(filename)
    print(f"Saved MIDI to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate Jazz Solos with RNN')
    parser.add_argument('--checkpoint', type=str, default='models/rnn/rnn_epoch_50.pth', help='Path to model checkpoint')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--output', type=str, default='output/midi/generated_solo.mid', help='Output MIDI filename')
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
    # Infer num_dur_classes from the checkpoint to avoid size mismatches when
    # the dataset's duration vocabulary has grown since the checkpoint was saved.
    num_dur_classes = dataset.vocab_size_duration
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'dur_embed.weight' in ckpt:
            num_dur_classes = ckpt['dur_embed.weight'].shape[0]
            print(f"Detected num_dur_classes={num_dur_classes} from checkpoint.")
    
    print("Initializing Model...")
    model = JazzRNN(num_dur_classes=num_dur_classes)
    
    # 3. Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(ckpt)
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")

    # 4. Initialize Generator
    generator = JazzGenerator(model, dataset)
    
    # 5. Define Chord Progression

    # Autumn Leaves
    progression = [
        (0, 1, 48),  # Cm7
        (5, 2, 48),  # F7
        (10,0, 48),  # Bbj7
        (3, 0, 48),  # Ebj7
        (9, 3, 48),  # Am7b5
        (2, 2, 48),  # D7
        (7, 1, 48),  # Gm7
        (7, 1, 48),  # Gm7

        (0, 1, 48),  # Cm7
        (5, 2, 48),  # F7
        (10,0, 48),  # Bbj7
        (3, 0, 48),  # Ebj7
        (9, 3, 48),  # Am7b5
        (2, 2, 48),  # D7
        (7, 1, 48),  # Gm7
        (7, 1, 48),  # Gm7

        (9, 3, 48),  # Am7b5
        (2, 2, 48),  # D7
        (7, 1, 48),  # Gm7
        (7, 1, 48),  # Gm7
        (0, 1, 48),  # Cm7
        (5, 2, 48),  # F7
        (10,0, 48),  # Bbj7
        (3, 0, 48),  # Ebj7

        (9, 3, 48),  # Am7b5
        (2, 2, 48),  # D7
        (7, 1, 24),  # Gm7
        (0, 2, 24),  # C7
        (5, 1, 24),  # Fm7
        (10,2, 24),  # Bb7
        (3, 0, 48),  # Ebj7
        (9, 3, 24),  # Am7b5
        (2, 2, 24),  # D7

        (7, 1, 48),  # Gm7
        (7, 1, 48),  # Gm7
    ]
    """
    # There will never be another you
    progression = [
        (3, 0, 48),  # Ebj7
        (3, 0, 48),  # Ebj7
        (2, 3, 48),  # Dm7b5
        (7, 2, 48),  # G7
        (0, 1, 48),  # Cm7
        (0, 1, 48),  # Cm7
        (10,1, 48),  # Bbm7
        (3, 2, 48),  # Eb7

        (8, 0, 48),  # Abj7
        (1, 2, 48),  # Db7
        (3, 0, 48),  # Ebj7
        (0, 1, 48),  # Cm7
        (5, 2, 48),  # F7
        (5, 2, 48),  # F7
        (5, 1, 48),  # Fm7
        (10,2, 48),  # Bb7

        (3, 0, 48),  # Ebj7
        (3, 0, 48),  # Ebj7
        (2, 3, 48),  # Dm7b5
        (7, 2, 48),  # G7
        (0, 1, 48),  # Cm7
        (0, 1, 48),  # Cm7
        (10,1, 48),  # Bbm7
        (3, 2, 48),  # Eb7
        
        (8, 0, 48),  # Abj7
        (1, 2, 48),  # Db7
        (3, 0, 48),  # Ebj7
        (8, 3, 24),  # Am7b5
        (2, 2, 24),  # D7
        (3, 0, 24),  # Ebj7
        (8, 2, 24),  # Ab7
        (7, 1, 24),  # Gm7
        (0, 2, 24),  # C7
        (5, 1, 24),  # Fm7
        (10,2, 24),  # Bb7
        (3, 0, 24),  # Ebj7
        (10,2, 24),  # Bb7
    ]
    # F Blues
    progression = [
        (5, 2, 48),  # F7
        (10,2, 48),  # Bb7
        (5, 2, 48),  # F7
        (5, 2, 48),  # F7
        (10,2, 48),  # Bb7
        (10,2, 48),  # Bb7
        (5, 2, 48),  # F7
        (9, 1, 24),  # Am7
        (2, 2, 24),  # D7
        (7, 1, 48),  # Gm7
        (0, 2, 48),  # C7
        (5, 2, 24),  # F7
        (2, 1, 24),  # Dm7
        (7, 1, 24),  # Gm7
        (0, 2, 24),  # C7
    ]
    """
    print("\nGenerating solo...")
    melody = generator.generate(progression, temperature=args.temp)
    
    save_to_midi(melody, args.output, args.bpm)

if __name__ == "__main__":
    main()
