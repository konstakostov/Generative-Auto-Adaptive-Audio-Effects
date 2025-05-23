from Effects.Delay_Effect.delay_effect import process_data_delay_effect
from Metadata.delay_metadata_generate import extract_delayed_metadata, cleanup_multistate_metadata_files

from path_constants import (
    CLEAN_AUDIO_DIR,
    DELAYED_AUDIO_DIR,
    METADATA_DELAY_DIR,
    METADATA_DELAY_OUTPUT,
)

import os


def apply_delay_effect(clean_audio_dir, delayed_audio_dir, metadata_delay_dir):
    """
    Applies a delay effect to clean audio files and generates metadata.
    """
    # Ensure parent DATA_DIR exists first
    os.makedirs(os.path.dirname(delayed_audio_dir), exist_ok=True)

    # Ensure the delayed audio directory exists
    os.makedirs(delayed_audio_dir, exist_ok=True)

    # Apply the delay effect to the clean audio files
    process_data_delay_effect(clean_audio_dir, delayed_audio_dir)
    print("\nDelay_Effect effect applied and saved to output directory.")

    # Create the metadata directory (not the file)
    os.makedirs(os.path.dirname(METADATA_DELAY_OUTPUT), exist_ok=True)

    # Extract metadata for the delayed audio files - pass the file path, not directory
    extract_delayed_metadata(delayed_audio_dir, METADATA_DELAY_OUTPUT)

    # Clean up intermediate metadata files
    cleanup_multistate_metadata_files(delayed_audio_dir)
    print("Metadata generated.")


# Apply the delay effect and generate metadata using predefined constants
apply_delay_effect(CLEAN_AUDIO_DIR, DELAYED_AUDIO_DIR, METADATA_DELAY_DIR)
