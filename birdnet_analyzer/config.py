import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

#################
# Misc settings #
#################

# Random seed for gaussian noise
RANDOM_SEED: int = 42

##########################
# Model paths and config #
##########################

MODEL_VERSION: str = "V2.4"
PB_MODEL: str = os.path.join(SCRIPT_DIR, "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model")
# MODEL_PATH = PB_MODEL # This will load the protobuf model
MODEL_PATH: str = os.path.join(SCRIPT_DIR, "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
MDATA_MODEL_PATH: str = os.path.join(SCRIPT_DIR, "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite")
LABELS_FILE: str = os.path.join(SCRIPT_DIR, "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt")
TRANSLATED_LABELS_PATH: str = os.path.join(SCRIPT_DIR, "labels/V2.4")

##################
# Audio settings #
##################

# We use a sample rate of 48kHz, so the model input size is
# (batch size, 48000 kHz * 3 seconds) = (1, 144000)
# Recordings will be resampled automatically.
SAMPLE_RATE: int = 48000

# We're using 3-second chunks
SIG_LENGTH: float = 3.0

# Define overlap between consecutive chunks <3.0; 0 = no overlap
SIG_OVERLAP: float = 0

# Define minimum length of audio chunk for prediction,
# chunks shorter than 3 seconds will be padded with zeros
SIG_MINLEN: float = 1.0

# Frequency range. This is model specific and should not be changed.
SIG_FMIN: int = 0
SIG_FMAX: int = 15000

# Settings for bandpass filter
BANDPASS_FMIN: int = 0
BANDPASS_FMAX: int = 15000

# Top N species to display in selection table, ignored if set to None
TOP_N = None

# Audio speed
AUDIO_SPEED: float = 1.0

#####################
# Metadata settings #
#####################

LATITUDE: float = -1
LONGITUDE: float = -1
WEEK: int = -1
LOCATION_FILTER_THRESHOLD: float = 0.03

######################
# Inference settings #
######################

# If None or empty file, no custom species list will be used
# Note: Entries in this list have to match entries from the LABELS_FILE
# We use the 2024 eBird taxonomy for species names (Clements list)
CODES_FILE: str = os.path.join(SCRIPT_DIR, "eBird_taxonomy_codes_2024E.json")
SPECIES_LIST_FILE: str = os.path.join(SCRIPT_DIR, "example/species_list.txt")

# Supported file types
ALLOWED_FILETYPES: list[str] = ["wav", "flac", "mp3", "ogg", "m4a", "wma", "aiff", "aif"]

# Number of threads to use for inference.
# Can be as high as number of CPUs in your system
CPU_THREADS: int = 8
TFLITE_THREADS: int = 1

# False will output logits, True will convert to sigmoid activations
APPLY_SIGMOID: bool = True
SIGMOID_SENSITIVITY: float = 1.0

# Minimum confidence score to include in selection table
# (be aware: if APPLY_SIGMOID = False, this no longer represents
# probabilities and needs to be adjusted)
MIN_CONFIDENCE: float = 0.25

# Number of consecutive detections for one species to merge into one
# If set to 1 or 0, no merging will be done
# If set to None, all detections will be included
MERGE_CONSECUTIVE: int = 1

# Number of samples to process at the same time. Higher values can increase
# processing speed, but will also increase memory usage.
# Might only be useful for GPU inference.
BATCH_SIZE: int = 1


# Number of seconds to load from a file at a time
# Files will be loaded into memory in segments that are only as long as this value
# Lowering this value results in lower memory usage
FILE_SPLITTING_DURATION: int = 600

# Whether to use noise to pad the signal
# If set to False, the signal will be padded with zeros
USE_NOISE: bool = False

# Specifies the output format. 'table' denotes a Raven selection table,
# 'audacity' denotes a TXT file with the same format as Audacity timeline labels
# 'csv' denotes a generic CSV file with start, end, species and confidence.
RESULT_TYPES: set[str] | list[str] = {"table"}
ADDITIONAL_COLUMNS: list[str] | None = None
OUTPUT_RAVEN_FILENAME: str = "BirdNET_SelectionTable.txt"  # this is for combined Raven selection tables only
# OUTPUT_RTABLE_FILENAME: str = "BirdNET_RTable.csv"
OUTPUT_KALEIDOSCOPE_FILENAME: str = "BirdNET_Kaleidoscope.csv"
OUTPUT_CSV_FILENAME: str = "BirdNET_CombinedTable.csv"

# File name of the settings csv for batch analysis
ANALYSIS_PARAMS_FILENAME: str = "BirdNET_analysis_params.csv"

# Whether to skip existing results in the output path
# If set to False, existing files will not be overwritten
SKIP_EXISTING_RESULTS: bool = False

COMBINE_RESULTS: bool = False
#####################
# Training settings #
#####################

# Sample crop mode
SAMPLE_CROP_MODE: str = "center"

# List of non-event classes
NON_EVENT_CLASSES: list[str] = ["noise", "other", "background", "silence"]

# Upsampling settings
UPSAMPLING_RATIO: float = 0.0
UPSAMPLING_MODE = "repeat"

# Number of epochs to train for
TRAIN_EPOCHS: int = 50

# Batch size for training
TRAIN_BATCH_SIZE: int = 32

# Validation split (percentage)
TRAIN_VAL_SPLIT: float = 0.2

# Learning rate for training
TRAIN_LEARNING_RATE: float = 0.0001

# Number of hidden units in custom classifier
# If >0, a two-layer classifier will be trained
TRAIN_HIDDEN_UNITS: int = 0

# Dropout rate for training
TRAIN_DROPOUT: float = 0.0

# Whether to use mixup for training
TRAIN_WITH_MIXUP: bool = False

# Whether to apply label smoothing for training
TRAIN_WITH_LABEL_SMOOTHING: bool = False

# Whether to use focal loss for training
TRAIN_WITH_FOCAL_LOSS: bool = False

# Focal loss gamma parameter
FOCAL_LOSS_GAMMA: float = 2.0

# Focal loss alpha parameter
FOCAL_LOSS_ALPHA: float = 0.25

# Model output format
TRAINED_MODEL_OUTPUT_FORMAT: str = "tflite"

# Model save mode (replace or append new classifier)
TRAINED_MODEL_SAVE_MODE: str = "replace"

# Cache settings
TRAIN_CACHE_MODE: str | None = None
TRAIN_CACHE_FILE: str = "train_cache.npz"

# Use automatic Hyperparameter tuning
AUTOTUNE: bool = False

# How many trials are done for the hyperparameter tuning
AUTOTUNE_TRIALS: int = 50

# How many executions per trial are done for the hyperparameter tuning
# Mutliple executions will be averaged, so the evaluation is more consistent
AUTOTUNE_EXECUTIONS_PER_TRIAL: int = 1

# If a binary classification model is trained.
# This value will be detected automatically in the training script, if only one class and a non-event class is used.
BINARY_CLASSIFICATION: bool = False

# If a model for a multi-label setting is trained.
# This value will automatically be set, if subfolders in the input direcotry are named with multiple classes separated by commas.
MULTI_LABEL: bool = False

################
# Runtime vars #
################

# File input path and output path for selection tables
INPUT_PATH: str = ""
OUTPUT_PATH: str = ""

# Training data path
TRAIN_DATA_PATH: str = ""
TEST_DATA_PATH: str = ""

CODES = {}
LABELS: list[str] = []
TRANSLATED_LABELS: list[str] = []
SPECIES_LIST: list[str] = []
ERROR_LOG_FILE: str = os.path.join(SCRIPT_DIR, "error_log.txt")
FILE_LIST = []
FILE_STORAGE_PATH: str = ""

# Path to custom trained classifier
# If None, no custom classifier will be used
# Make sure to set the LABELS_FILE above accordingly
CUSTOM_CLASSIFIER: str | None = None

######################
# Get and set config #
######################


def get_config():
    return {k: v for k, v in globals().items() if k.isupper()}


def set_config(c: dict):
    for k, v in c.items():
        globals()[k] = v
