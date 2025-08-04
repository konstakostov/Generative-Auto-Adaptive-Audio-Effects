from Effects.Delay_Effect.delay_effect import process_data_delay_effect
from Metadata.delay_metadata_generate import extract_delayed_metadata, cleanup_multistate_metadata_files

from Utils.path_constants import (
    CLEAN_AUDIO_DIR,
    DELAYED_AUDIO_DIR,
    METADATA_DELAY_DIR,
)
from Utils.unique_naming_delay import create_unique_delay_subfolder_id

import os


def apply_delay_effect(clean_audio_dir, delayed_audio_dir, metadata_delay_dir):
    print(f"Looking for audio files in: {clean_audio_dir}")

    # Ensure all parent directories exist
    os.makedirs(clean_audio_dir, exist_ok=True)
    os.makedirs(delayed_audio_dir, exist_ok=True)
    os.makedirs(metadata_delay_dir, exist_ok=True)

    # Create unique output folder
    unique_delayed_audio_dir, timestamp = create_unique_delay_subfolder_id(delayed_audio_dir)
    print(f"Output will be saved to: {unique_delayed_audio_dir}")

    # Process audio files
    process_data_delay_effect(clean_audio_dir, unique_delayed_audio_dir)
    print("\nDelay_Effect effect applied and saved to output directory.")

    # Generate metadata
    metadata_output_path = os.path.join(metadata_delay_dir, f"delay_metadata_{timestamp}.json")
    extract_delayed_metadata(unique_delayed_audio_dir, metadata_output_path)
    cleanup_multistate_metadata_files(unique_delayed_audio_dir)
    print("Metadata generated.")


apply_delay_effect(CLEAN_AUDIO_DIR, DELAYED_AUDIO_DIR, METADATA_DELAY_DIR)
