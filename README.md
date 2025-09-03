# Generative-Auto-Adaptive-Audio-Effects

# Project Setup and Usage

## Prerequisites

- Python (recommended version: 3.8+)
- pip (Python package manager)

## Installation

Clone the repository and install dependencies: ```pip install -r requirements.txt```

## Data Folder Setup

The `Data` folder is **not** included in the repository. You must create it manually in the project root directory. Inside Data, create the following subfolders:
```
<Root>/
├── Data/
│   ├── Audio_Inputs/
│   ├── Audio_Outputs/
│   └── Clean_Audio_Samples/
│   └── Delayed_Audio_Samples/
│   └── Text_Files/
```

Place your data files in the appropriate subfolders: 
- `Audio_Inputs/` --> Existing `.wav` files that can be used for applying the audio effect.
- `Audio_Outputs/` --> Proccessed `.wav` files with an audio effect.
- `Clean_Audio_Samples/` --> Audio Samples (in `.wav`) from a dataset. In this project the **EGFxSet**'s clean files has been used. Link to the dataset is provided at the bottom of the `README.md`.
- `Delayed_Audio_Samples/` --> The dataset files with the applied delayed effect, used for creating the transition matrix for the Markov Chain by the simple NN.
- `Text_Files/` --> Print statements giving insight of a processed audio input with an audio effect for a given time frame.

## Running the Project

From the `main.py` file the project can be run by (un)commenting desired sections: 
- `apply_delay_effect` Applies delay effect to the clean files from the dataset, saves them as processed files in the provided directory, and creates metadata to be used by the NN for transition matrix creation.
  - To create the transition matrix required for the Markov Chain, run `delay_nn.ipynb` from the `Notebooks\` directory.
- `process_mic_input` Processes audio input from a mic and applies the desired audio effect.
- `process_wav_input` Processes audio input from a `.wav` file and applies the desired audio effect.
- The results from the processed audio files are saved to a `.txt` file.
  
## Notes

Ensure the `Data` folder and its subfolders exist before running the project. If you encounter errors related to missing data, verify the folder structure and file locations.

## Sources
- [EGFxSet Paper](https://www.researchgate.net/publication/388360501_EGFXSET_ELECTRIC_GUITAR_TONES_PROCESSED_THROUGH_REAL_EFFECTS_OF_DISTORTION_MODULATION_DELAY_AND_REVERB)
- [EGFxSet Dataset](https://egfxset.github.io)
