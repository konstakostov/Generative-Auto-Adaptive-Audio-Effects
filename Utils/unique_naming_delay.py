import os
from datetime import datetime


def create_unique_delay_subfolder_id(base_dir):
    """
    Create a unique subfolder for storing delayed audio samples.

    Parameters:
        base_dir (str): The base directory where the subfolder will be created.

    Returns:
        tuple: A tuple containing:
            - folder_id (str): The full path to the created subfolder.
            - timestamp (str): The timestamp used in the subfolder name.

    Notes:
        - The subfolder name is generated using the current timestamp in the format 'YYYY.MM.DD.HH.MM.SS'.
        - If the subfolder already exists, it will not raise an error due to `exist_ok=True`.
    """
    # Generate a timestamp in the format 'YYYY.MM.DD.HH.MM.SS'
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Create the subfolder name using the timestamp
    folder_name = f"Delayed_Audio_Samples_{timestamp}"
    folder_id = os.path.join(base_dir, folder_name)

    # Create the subfolder, ensuring it exists
    os.makedirs(folder_id, exist_ok=True)

    # Return the full path to the subfolder and the timestamp
    return folder_id, timestamp