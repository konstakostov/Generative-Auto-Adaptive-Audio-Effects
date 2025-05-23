import os


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

CLEAN_AUDIO_DIR = os.path.join(DATA_DIR, "Clean_Audio_Samples")
DELAYED_AUDIO_DIR = os.path.join(DATA_DIR, "Delayed_Audio_Samples")


EFFECTS_DIR = os.path.join(PROJECT_ROOT, "Effects")
EFFECTS_DELAY_DIR = os.path.join(EFFECTS_DIR, "Delay_Effect")


METADATA_DIR = os.path.join(PROJECT_ROOT, "Metadata")
METADATA_DELAY_DIR = os.path.join(METADATA_DIR, "Delay_Metadata")
METADATA_DELAY_OUTPUT = os.path.join(METADATA_DELAY_DIR, "delay_metadata.json")


MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
MODELS_DELAY_DIR = os.path.join(MODELS_DIR, "Delay_Models")


NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "Notebooks")
NOTEBOOKS_DELAY_DIR = os.path.join(NOTEBOOKS_DIR, "Delay_Notebooks")


MATRICES_DIR = os.path.join(PROJECT_ROOT, "Matrices")
MATRICES_DELAY_DIR = os.path.join(MATRICES_DIR, "Matrices_Delay")
