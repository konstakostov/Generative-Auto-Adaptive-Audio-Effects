import os
from datetime import datetime


def create_unique_delay_subfolder_id(base_dir):
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    folder_name = f"Delayed_Audio_Samples_{timestamp}"
    folder_id = os.path.join(base_dir, folder_name)

    os.makedirs(folder_id, exist_ok=True)

    return folder_id, timestamp