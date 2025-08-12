import os

# Define the root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define the directory for storing data
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Directory for storing audio input files
AUDIO_INPUTS_DIR = os.path.join(DATA_DIR, "Audio_Inputs")

# Directory for storing audio output files
AUDIO_OUTPUTS_DIR = os.path.join(DATA_DIR, "Audio_Outputs")

# Directory for storing delayed audio output files
DELAY_AUDIO_OUTPUTS_DIR = os.path.join(AUDIO_OUTPUTS_DIR, "Delay_Audio_Outputs")

# Directory for storing delayed microphone output files
DELAY_MIC_OUTPUTS_DIR = os.path.join(DELAY_AUDIO_OUTPUTS_DIR, "Delay_Mic_Outputs")

# Directory for storing delayed WAV output files
DELAY_WAV_OUTPUTS_DIR = os.path.join(DELAY_AUDIO_OUTPUTS_DIR, "Delay_Wav_Outputs")

# Define the directory for storing clean audio samples
CLEAN_AUDIO_DIR = os.path.join(DATA_DIR, "Clean_Audio_Samples")

# Define the directory for storing delayed audio samples
DELAYED_AUDIO_DIR = os.path.join(DATA_DIR, "Delayed_Audio_Samples")

# Directory for storing text files
TEXT_FILES_DIR = os.path.join(DATA_DIR, "Text_Files")

# Directory for storing delayed audio text files
DELAY_TEXT_FILES_DIR = os.path.join(TEXT_FILES_DIR, "Delay_Text_Files")

# Define the directory for storing audio effects
EFFECTS_DIR = os.path.join(PROJECT_ROOT, "Audio_Effects")

# Define the directory for storing delay effect files
EFFECTS_DELAY_DIR = os.path.join(EFFECTS_DIR, "Delay_Effect")

# Define the directory for storing metadata
METADATA_DIR = os.path.join(PROJECT_ROOT, "Metadata")

# Define the directory for storing delay metadata
METADATA_DELAY_DIR = os.path.join(METADATA_DIR, "Delay_Metadata")

# Define the directory for storing models
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")

# Define the directory for storing delay models
MODELS_DELAY_DIR = os.path.join(MODELS_DIR, "Delay_Models")

# Define the directory for storing notebooks
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "Notebooks")

# Define the directory for storing delay-related notebooks
NOTEBOOKS_DELAY_DIR = os.path.join(NOTEBOOKS_DIR, "Delay_Notebooks")

# Define the directory for storing matrices
MATRICES_DIR = os.path.join(PROJECT_ROOT, "Matrices")

# Define the directory for storing delay-related matrices
MATRICES_DELAY_DIR = os.path.join(MATRICES_DIR, "Matrices_Delay")
