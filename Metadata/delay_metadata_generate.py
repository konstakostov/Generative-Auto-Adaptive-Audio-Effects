import os
import json

import numpy as np
import soundfile as sf
from librosa import zero_crossings
from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_flatness, mfcc
from tqdm import tqdm


def extract_delayed_metadata(audio_dir, output_file, n_mfcc=13):
    """
    Extracts metadata from delayed audio samples and saves it to a JSON file.

    Args:
        audio_dir (str): Path to the directory containing delayed audio samples.
        output_file (str): Path to the output JSON file where metadata will be saved.
        n_mfcc (int, optional): Number of MFCC (Mel-frequency cepstral coefficients) to compute. Defaults to 13.

    The function processes all `*_multistate_metadata.json` files in the given directory,
    extracts audio features for each segment, and generates metadata entries with details
    such as RMS, zero-crossing rate, spectral features, and MFCCs. The metadata is saved
    to the specified output file.
    """

    metadata = []

    # Find all multi-state metadata files
    print("Processing multi-state files...")
    metadata_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith("_multistate_metadata.json"):
                metadata_files.append(os.path.join(root, file))

    # Process multi-state files and create transition entries
    for metadata_file in tqdm(metadata_files, desc="Processing multi-state files"):
        try:
            with open(metadata_file, 'r') as f:
                file_meta = json.load(f)

            # Construct the corresponding audio file path
            audio_file = os.path.join(
                str(os.path.dirname(metadata_file)),
                str(os.path.basename(metadata_file).replace('_metadata.json', '.wav'))
            )

            # Skip if the audio file does not exist
            if not os.path.exists(audio_file):
                print(f"Audio file not found: {audio_file}")
                continue

            # Read the audio file and extract features
            audio, sr = sf.read(audio_file)
            mono_audio = audio if len(audio.shape) == 1 else audio[:, 0]  # Use mono channel
            segment_len = len(mono_audio) // file_meta["segment_count"]

            # For each segment, create a metadata entry with next_state info
            sequence = file_meta["sequence"]
            for i, segment_info in enumerate(sequence):
                start = i * segment_len
                end = start + segment_len if i < file_meta["segment_count"] - 1 else len(mono_audio)
                segment = mono_audio[start:end]

                # Extract features for this segment
                rms = float(np.sqrt(np.mean(segment ** 2)))
                zero_crossings_feature = float(np.mean(zero_crossings(segment, pad=False)))
                spectral_centroid_feature = float(spectral_centroid(y=segment, sr=sr).mean())
                spectral_bandwidth_feature = float(spectral_bandwidth(y=segment, sr=sr).mean())
                spectral_flatness_feature = float(spectral_flatness(y=segment).mean())
                peak_amplitude_feature = float(np.max(np.abs(segment)))
                mfccs_feature = mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean_feature = mfccs_feature.mean(axis=1).tolist()
                mfcc_std_feature = mfccs_feature.std(axis=1).tolist()

                # Create metadata entry
                entry = {
                    "relative_path": os.path.relpath(audio_file, audio_dir),
                    "duration": len(segment) / sr,
                    "sample_rate": sr,
                    "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
                    "state": segment_info["state"],
                    "delay_time_ms": segment_info["delay_time"] * 1000,
                    "segment_index": segment_info["segment_index"],
                    "rms": rms,
                    "zero_crossing_rate": zero_crossings_feature,
                    "spectral_centroid": spectral_centroid_feature,
                    "spectral_bandwidth": spectral_bandwidth_feature,
                    "spectral_flatness": spectral_flatness_feature,
                    "peak_amplitude": peak_amplitude_feature,
                    "mfcc_mean": mfcc_mean_feature,
                    "mfcc_std": mfcc_std_feature,
                }

                # Add next_state field for all but the last segment
                if i < len(sequence) - 1:
                    entry["next_state"] = sequence[i + 1]["state"]

                metadata.append(entry)

        except Exception as e:
            print(f"Error processing multi-state file {metadata_file}: {e}")

    # Write all metadata to file
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def cleanup_multistate_metadata_files(directory):
    """
    Deletes all temporary `*_multistate_metadata.json` files in the given directory and its subdirectories.

    Args:
        directory (str): Path to the directory where temporary metadata files should be deleted.

    This function recursively searches for files ending with `_multistate_metadata.json`
    and deletes them. If a file cannot be deleted, an error message is printed.
    """

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_multistate_metadata.json"):
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")

    print("Temporary metadata files deleted.")
