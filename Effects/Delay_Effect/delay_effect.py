import os
import json

import numpy as np
import soundfile as sf
from tqdm import tqdm

from Effects.perlin_noise import generate_perlin_noise_sequence
from Effects.audio_effects import DelayEffect

# Set the number of segments to split the audio into
SEGMENT_COUNT = 10

# Define the delay times for different states
DELAY_STATES_TIMES = {
    # Short State (20ms - 79ms), Generated 31 values
    "short": [round(0.001 * x, 3) for x in range(20, 80, 2)],
    # Medium State (80ms - 349ms), Generated 28 values
    "medium": [round(0.001 * x, 3) for x in range(80, 350, 10)],
    # Long State (350ms - 1500ms), Generated 30 values
    "long": [round(0.001 * x, 3) for x in range(350, 1501, 40)],
}

# Thresholds for mapping Perlin noise values to delay states:
# Values below 0.33 map to "short", between 0.33 and 0.66 to "medium", and above 0.66 to "long".
STATE_THRESHOLDS = [0.33, 0.66]


def process_data_delay_effect(input_dir, output_dir):
    """
    Processes audio files by applying a multi-state delay effect and saves the processed audio
    along with metadata.

    Args:
        input_dir (str): Path to the directory containing input audio files.
        output_dir (str): Path to the directory where processed audio and metadata will be saved.

    Steps:
        1. Walks through the input directory to find `.wav` files.
        2. Splits each audio file into segments and applies a delay effect based on Perlin noise.
        3. Saves the processed audio and metadata for each file in the output directory.
    """
    # Collect all `.wav` audio files from the input directory
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                audio_files.append((root, file, full_path))

    # Process each audio file
    for root, file, full_path in tqdm(audio_files, desc="Processing audio files"):
        rel_path = os.path.relpath(str(full_path), str(input_dir))

        # Read the input audio file
        input_audio, sample_rate = sf.read(full_path)

        # Calculate segment length
        segment_len = len(input_audio) // SEGMENT_COUNT

        segments = []
        state_sequence = []

        # Generate Perlin noise values for segment state selection
        noise_values = generate_perlin_noise_sequence(SEGMENT_COUNT, scale=0.3, seed=None)

        for i in range(SEGMENT_COUNT):
            start = i * segment_len

            # Determine the end of the segment
            if i < SEGMENT_COUNT - 1:
                end = start + segment_len
            else:
                end = len(input_audio)

            # Extract the segment
            segment = input_audio[start:end]

            # Determine the state based on noise values
            noise_val = noise_values[i]
            if noise_val < STATE_THRESHOLDS[0]:
                state = "short"
            elif noise_val < STATE_THRESHOLDS[1]:
                state = "medium"
            else:
                state = "long"

            # Select a delay time for the current state
            state_times = DELAY_STATES_TIMES[state]
            delay_noise = generate_perlin_noise_sequence(SEGMENT_COUNT, scale=0.4, seed=None)
            delay_idx = int(delay_noise[i] * len(state_times)) % len(state_times)
            delay_time = state_times[delay_idx]

            # Apply the delay effect to the segment
            effect = DelayEffect(sample_rate=sample_rate, delay_time=delay_time, gain=0.5)
            processed_segment = effect.process(segment)
            segments.append(processed_segment)

            # Record the state and delay time for metadata
            state_sequence.append({
                "segment_index": i,
                "state": state,
                "delay_time": delay_time
            })

        # Concatenate all processed segments into a single audio file
        output_audio = np.concatenate(segments)

        # Create the output directory structure
        output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)

        # Save the processed audio file
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_filename = f"{base_name}_multistate.wav"
        output_path = os.path.join(output_subdir, output_filename)
        sf.write(output_path, output_audio, sample_rate)

        # Save metadata as a JSON file
        metadata_path = os.path.join(output_subdir, f"{base_name}_multistate_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "file": rel_path,
                "segment_count": SEGMENT_COUNT,
                "sequence": state_sequence
            }, f)
