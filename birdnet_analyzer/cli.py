# ruff: noqa: E501
import argparse
import os

import birdnet_analyzer.config as cfg

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ASCII_LOGO = r"""                        
                          .                                     
                       .-=-                                     
                    .:=++++.                                    
                 ..-======#=:.                                  
                .-%%%#*+=-#+++:..                               
              .-+***======++++++=..                             
                  .=====+==++++++++-.                           
                  .=+++====++++++++++=:.                        
                  .++++++++=======----===:                      
                   =+++++++====-----+++++++-.                   
                   .=++++==========-=++=====+=:.                
                     -++======---:::::-=++++***+:.              
                     ..---::::::::::::::::-=*****+-.            
                       ..--------::::::::::::--+##-.:.          
  ++++=::::::...         ..-------------::::::-::.::.           
           ..::-------:::.-=.:::::+-....   ....:--:..           
                    ..::-======--+::......      .:---:.         
                              ..:--==+++++==-..    .-+==-       
                                   ......::----:      **=--     
                                            ..-=-:.     *+=:=   
                                              ..-====  +++ =+** 
                                                 ========+      
                                                 **=====        
                                               ***+==           
                                              ****+             
"""  # noqa: W291


def io_args():
    """
    Creates an argument parser for input and output paths.
    Returns:
        argparse.ArgumentParser: The argument parser with input and output path arguments.
    Arguments:
        input (str): Path to the input file or folder. Defaults to the value specified in cfg.INPUT_PATH.
        output (str): Path to the output folder. Defaults to the input path if not specified.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "audio_input",
        metavar="INPUT",
        help="Path to input file or folder.",
    )
    p.add_argument("-o", "--output", help="Path to output folder. Defaults to the input path.")

    return p


def bandpass_args():
    """
    Creates an argument parser for bandpass filter frequency arguments.
    This function sets up an argument parser with two arguments:
    --fmin and --fmax, which define the minimum and maximum frequencies
    for the bandpass filter, respectively. The values are constrained
    to be within the range defined by cfg.SIG_FMIN and cfg.SIG_FMAX.
    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--fmin",
        type=lambda a: max(0, min(cfg.SIG_FMAX, int(a))),
        default=cfg.SIG_FMIN,
        help="Minimum frequency for bandpass filter in Hz.",
    )
    p.add_argument(
        "--fmax",
        type=lambda a: max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(a))),
        default=cfg.SIG_FMAX,
        help="Maximum frequency for bandpass filter in Hz.",
    )

    return p

def species_list_args():
    """
    Creates an argument parser for species-list arguments.
    Returns:
        argparse.ArgumentParser: The argument parser with the following arguments:
            --lat (float): Recording location latitude. Set -1 to ignore. Default is -1.
            --lon (float): Recording location longitude. Set -1 to ignore. Default is -1.
            --week (int): Week of the year when the recording was made. Values in [1, 48] (4 weeks per month).
                          Set -1 for year-round species list. Default is -1.
            --slist (str): Path to species list file or folder. If folder is provided, species list needs to be named
                           "species_list.txt". If lat and lon are provided, this list will be ignored.
            --sf_thresh (float): Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99].
                                 Defaults to cfg.LOCATION_FILTER_THRESHOLD.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lat", type=float, default=-1, help="Recording location latitude. Set -1 to ignore.")
    p.add_argument("--lon", type=float, default=-1, help="Recording location longitude. Set -1 to ignore.")
    p.add_argument(
        "--week",
        type=int,
        default=-1,
        help="Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.",
    )
    p.add_argument(
        "--sf_thresh",
        type=lambda a: max(0.0001, min(0.99, float(a))),
        default=cfg.LOCATION_FILTER_THRESHOLD,
        help="Minimum species occurrence frequency threshold for location filter. Values in [0.0001, 0.99].",
    )
    return p

def species_args():
    """
    Creates an argument parser for species-related arguments including the species-list arguments.
    Returns:
        argparse.ArgumentParser: The argument parser with the following arguments:
            --lat (float): Recording location latitude. Set -1 to ignore. Default is -1.
            --lon (float): Recording location longitude. Set -1 to ignore. Default is -1.
            --week (int): Week of the year when the recording was made. Values in [1, 48] (4 weeks per month).
                          Set -1 for year-round species list. Default is -1.
            --sf_thresh (float): Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99].
                                 Defaults to cfg.LOCATION_FILTER_THRESHOLD.
            --slist (str): Path to species list file or folder. If folder is provided, species list needs to be named
                           "species_list.txt". If lat and lon are provided, this list will be ignored.
    """
    p = species_list_args()
    p.add_argument(
        "--slist",
        help='Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.',
    )
    return p


def sigmoid_args():
    """
    Creates an argument parser for sigmoid sensitivity.
    This function sets up an argument parser with a single argument `--sensitivity`.
    The sensitivity value is constrained to be within the range [0.75, 1.25], where higher
    values result in higher detection sensitivity. The default value is taken from
    `cfg.SIGMOID_SENSITIVITY`.
    Returns:
        argparse.ArgumentParser: The argument parser with the sensitivity argument configured.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--sensitivity",
        type=lambda a: min(1.25, max(0.75, float(a))),
        default=cfg.SIGMOID_SENSITIVITY,
        help="Detection sensitivity; Higher values result in higher sensitivity. Values in [0.75, 1.25]. Values other than 1.0 will shift the sigmoid functionon the x-axis. Use complementary to the cut-off threshold.",
    )

    return p


def overlap_args(help_string="Overlap of prediction segments. Values in [0.0, 2.9]."):
    """
    Creates an argument parser for the overlap of prediction segments.
    Args:
        help_string (str): A custom help string for the overlap argument. Defaults to a formatted string
                           indicating the range [0.0, 2.9] and the default value from cfg.SIG_OVERLAP.
    Returns:
        argparse.ArgumentParser: An argument parser with the overlap argument configured.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--overlap",
        type=lambda a: max(0.0, min(2.9, float(a))),
        default=cfg.SIG_OVERLAP,
        help=help_string,
    )

    return p


def audio_speed_args():
    """
    Creates an argument parser for audio speed configuration.
    This function sets up an argument parser with a single argument `--audio_speed`
    which allows the user to specify a speed factor for audio playback. The speed factor
    must be a float value where values less than 1.0 will slow down the audio and values
    greater than 1.0 will speed it up. The minimum allowed value is 0.01. The default
    value is taken from the configuration (`cfg.AUDIO_SPEED`).
    Returns:
        argparse.ArgumentParser: The argument parser with the `--audio_speed` argument configured.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--audio_speed",
        type=lambda a: max(0.01, float(a)),
        default=cfg.AUDIO_SPEED,
        help="Speed factor for audio playback. Values < 1.0 will slow down the audio, values > 1.0 will speed it up. At a 10x decrease (audio speed 0.1), a 384 kHz recording becomes a 38.4 kHz recording.",
    )

    return p


def threads_args():
    """
    Creates an argument parser for specifying the number of CPU threads to use.
    The parser adds an argument `--threads` (or `-t`) which accepts an integer value.
    The value is constrained to be at least 1. If not specified, the default value is
    set to half the number of available CPU cores, but not exceeding 8.
    Returns:
        argparse.ArgumentParser: The argument parser with the `--threads` argument.
    """
    import multiprocessing

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-t",
        "--threads",
        type=lambda a: max(1, int(a)),
        default=min(8, max(1, multiprocessing.cpu_count() // 2)),
        help="Number of CPU threads.",
    )

    return p


def min_conf_args():
    """
    Creates an argument parser for the minimum confidence threshold.

    Returns:
        argparse.ArgumentParser: An argument parser with the --min_conf argument.

    The --min_conf argument:
        - Sets the minimum confidence threshold for predictions.
        - Accepts float values in the range [0.01, 0.99].
        - Defaults to the value specified in cfg.MIN_CONFIDENCE.
        - Ensures that the provided value is clamped between 0.01 and 0.99.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--min_conf",
        default=cfg.MIN_CONFIDENCE,
        type=lambda a: max(0.00001, min(0.99, float(a))),
        help="Minimum confidence threshold. Values in [0.00001, 0.99].",
    )

    return p


def locale_args():
    """
    Creates an argument parser for locale settings.
    This function creates an argument parser with a single argument `--locale`
    (or `-l`) which specifies the locale for translated species common names.
    The default value is 'en' (US English). The available locale values include
    'af', 'en_UK', 'de', 'it', and others.
    Returns:
        argparse.ArgumentParser: An argument parser with the locale argument.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-l",
        "--locale",
        default="en",
        help="Locale for translated species common names. Values in ['af', 'en_UK', 'de', 'it', ...].",
    )

    return p


def bs_args(default=cfg.BATCH_SIZE):
    """
    Creates an argument parser for batch size configuration.
    Returns:
        argparse.ArgumentParser: An argument parser with a batch size argument.
    The parser includes the following argument:
        -b, --batch_size: An integer specifying the number of samples to process at the same time.
                         The value must be at least 1. Defaults to the value of cfg.BATCH_SIZE.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-b",
        "--batch_size",
        type=lambda a: max(1, int(a)),
        default=default,
        help="Number of samples to process at the same time.",
    )

    return p


def db_args():
    """
    Creates an arguments parser for the database path.
    Returns:
        argparse.ArgumentParser: An argument parser with a database size argument.
    The parser includes the following argument:
        -db, --database: Path to the database folder.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-db",
        "--database",
        help="Path to the database folder.",
    )

    return p


def analyzer_parser():
    """
    Creates and returns an argument parser for the BirdNET Analyzer CLI.
    The parser includes various argument groups for different functionalities such as
    I/O operations, bandpass filtering, species selection, sigmoid function parameters,
    overlap settings, audio speed adjustments, threading, minimum confidence levels,
    locale settings, and batch size.
    If the environment variable "IS_GITHUB_RUNNER" is set to "true", a simplified parser
    description is used. Otherwise, a detailed ASCII logo and usage instructions are included.
    The parser also defines a custom action `UniqueSetAction` to ensure that the `--rtype`
    argument values are stored as a set of unique, lowercase strings.
    Arguments:
        --rtype: Specifies output format. Accepts multiple values from ['table', 'audacity', 'kaleidoscope', 'csv'].
        --combine_results: Outputs a combined file for all selected result types if set.
        -c, --classifier: Path to a custom trained classifier. Overrides --lat, --lon, and --locale if set.
        --skip_existing_results: Skips files that have already been analyzed if set.
        --top_n: Saves only the top N predictions for each segment. Threshold will be ignored.
        --merge_consecutive: Maximum number of consecutive detections to merge for each species.
    Returns:
        argparse.ArgumentParser: Configured argument parser for the BirdNET Analyzer CLI.
    """
    from birdnet_analyzer.analyze import POSSIBLE_ADDITIONAL_COLUMNS_MAP
    parents = [
        io_args(),
        bandpass_args(),
        species_args(),
        sigmoid_args(),
        overlap_args(),
        audio_speed_args(),
        threads_args(),
        min_conf_args(),
        locale_args(),
        bs_args(),
    ]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
    )

    class UniqueSetAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            setattr(args, self.dest, {v.lower() for v in values})

    parser.add_argument(
        "--rtype",
        default={"table"},
        choices=["table", "audacity", "kaleidoscope", "csv"],
        nargs="+",
        help="Specifies output format. Values in `['table', 'audacity',  'kaleidoscope', 'csv']`.",
        action=UniqueSetAction,
    )
    parser.add_argument(
        "--additional_columns",
        choices=POSSIBLE_ADDITIONAL_COLUMNS_MAP.keys(),
        nargs="+",
        help="Additional columns to include in the output, only applied to the csv output format. "
        + f"Values in [{','.join(POSSIBLE_ADDITIONAL_COLUMNS_MAP.keys())}].",
        action=UniqueSetAction,
    )
    parser.add_argument(
        "--combine_results",
        help="Also outputs a combined file for all the selected result types. If not set combined tables will be generated.",
        action="store_true",
    )

    parser.add_argument(
        "-c",
        "--classifier",
        default=cfg.CUSTOM_CLASSIFIER,
        help="Path to custom trained classifier. If set, --lat, --lon and --locale are ignored.",
    )

    parser.add_argument(
        "--skip_existing_results",
        action="store_true",
        help="Skip files that have already been analyzed.",
    )

    parser.add_argument(
        "--top_n",
        type=lambda a: max(1, int(a)),
        help="Saves only the top N predictions for each segment independent of their score. Threshold will be ignored.",
    )

    parser.add_argument(
        "--merge_consecutive",
        type=int,
        default=1,
        help="Maximum number of consecutive detections above MIN_CONF to merge for each detected species. This will result in fewer entires in the result file with segments longer than 3 seconds. Set to 0 or 1 to disable merging. Set to None to include all consecutive detections. We use the mean of the top 3 scores from all consecutive detections for merging.",
    )

    return parser


def embeddings_parser():
    """
    Creates and returns an argument parser for extracting feature embeddings with BirdNET.

    The parser includes arguments from the following parent parsers:
    - db_args(): Handles database arguments.
    - bandpass_args(): Handles bandpass filter arguments.
    - audio_speed_args(): Handles audio speed arguments.
    - overlap_args(): Handles overlap arguments.
    - threads_args(): Handles threading arguments.
    - bs_args(): Handles batch size arguments.

    Returns:
        argparse.ArgumentParser: Configured argument parser for extracting feature embeddings.
    """

    parents = [db_args(), bandpass_args(), audio_speed_args(), overlap_args(), threads_args(), bs_args()]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="audio_input",
        help="Path to input file or folder.",
    )

    parser.add_argument(
        "--file_output",
    )

    return parser


def search_parser():
    """
    Creates and returns an argument parser for searching BirdNET embeddings.

    The parser includes the following arguments:
    - -q, --queryfile: Path to the query file.
    - -o, --output: Path to the output folder.
    - --n_results: Number of results to return.
    - --score_function: Scoring function to use. Choose 'cosine', 'euclidean' or 'dot'. Defaults to 'cosine'.
    - --crop_mode: Crop mode for the query sample. Can be 'center', 'first' or 'segments'.

    The parser also includes arguments from the following parent parsers:
    - overlap_args(): Handles overlap arguments if segments is selected as crop mode.
    - db_args(): Handles database arguments.
    """

    parents = [overlap_args(), db_args()]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=parents)
    parser.add_argument("-q", "--queryfile", help="Path to the query file.")
    parser.add_argument("-o", "--output", help="Path to the output folder.")
    parser.add_argument("--n_results", default=10, type=int, help="Number of results to return.")

    # TODO: use choice argument.
    parser.add_argument(
        "--score_function",
        default="cosine",
        choices=["cosine", "euclidean", "dot"],
        help="Scoring function to use. Choose 'cosine', 'euclidean' or 'dot'. Defaults to 'cosine'.",
    )
    parser.add_argument(
        "--crop_mode",
        default=cfg.SAMPLE_CROP_MODE,
        choices=["center", "first", "segments"],
        help="Crop mode for the query sample. Can be 'center', 'first' or 'segments'.",
    )

    return parser


def client_parser():
    """
    Creates and returns an argument parser for the client that queries an analyzer API endpoint server.
    The parser includes the following arguments:
    - --host: Host name or IP address of the API endpoint server (default: "localhost").
    - -p, --port: Port of the API endpoint server (default: 8080).
    - --pmode: Score pooling mode, with possible values 'avg' or 'max' (default: "avg").
    - --num_results: Number of results per request (default: 5).
    - --save: Flag to define if files should be stored on the server.
    The parser also includes arguments from the following parent parsers:
    - io_args()
    - species_args()
    - sigmoid_args()
    - overlap_args()
    Returns:
        argparse.ArgumentParser: Configured argument parser for the client.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[io_args(), species_args(), sigmoid_args(), overlap_args()],
    )
    parser.add_argument("--host", default="localhost", help="Host name or IP address of API endpoint server.")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port of API endpoint server.")
    parser.add_argument("--pmode", default="avg", help="Score pooling mode. Values in ['avg', 'max'].")
    parser.add_argument("--num_results", type=int, default=5, help="Number of results per request.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Define if files should be stored on server.",
    )

    return parser


def segments_parser():
    """
    Creates an argument parser for extracting segments from audio files based on BirdNET detections.
    Returns:
        argparse.ArgumentParser: Configured argument parser with the following arguments:
            - input (str): Path to folder containing audio files.
            - results (str, optional): Path to folder containing result files. Defaults to the `input` path.
            - output (str, optional): Output folder path for extracted segments. Defaults to the `input` path.
            - max_segments (int, optional): Number of randomly extracted segments per species. Defaults to 100.
            - seg_length (float, optional): Length of extracted segments in seconds. Defaults to cfg.SIG_LENGTH.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[audio_speed_args(), threads_args(), min_conf_args()],
    )
    parser.add_argument("audio_input", metavar="INPUT", help="Path to folder containing audio files.")
    parser.add_argument("-r", "--results", help="Path to folder containing result files. Defaults to the `input` path.")
    parser.add_argument("-o", "--output", help="Output folder path for extracted segments. Defaults to the `input` path.")
    parser.add_argument(
        "--max_segments",
        type=lambda a: max(1, int(a)),
        default=100,
        help="Number of randomly extracted segments per species.",
    )

    parser.add_argument(
        "--seg_length",
        type=lambda a: max(1.0, float(a)),
        default=cfg.SIG_LENGTH,
        help="Minimum length of extracted segments in seconds. If a segment is shorter than this value, it will be padded with audio from the source file.",
    )

    return parser


def server_parser():
    """
    Creates and configures an argument parser for the API endpoint server.
    The parser includes arguments for specifying the host, port, and storage path for uploaded files.
    It also inherits arguments from `threads_args` and `locale_args`.
    Returns:
        argparse.ArgumentParser: Configured argument parser with server-specific options.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[threads_args(), locale_args()],
    )

    parser.add_argument("--host", default="0.0.0.0", help="Host name or IP address of API endpoint server.")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port of API endpoint server.")
    parser.add_argument(
        "--spath",
        default="uploads/" if os.environ.get("IS_GITHUB_RUNNER", "false").lower() == "true" else os.path.join(SCRIPT_DIR, "uploads"),
        help="Path to folder where uploaded files should be stored.",
    )

    return parser


def species_parser():
    """
    Creates an argument parser for retrieving a list of species for a given location using BirdNET.
    The parser includes the following arguments:
    - output: Path to the output file or folder. If a folder is provided, the file will be named 'species_list.txt'.
    - --sortby: Optional argument to sort species by occurrence frequency ('freq') or alphabetically ('alpha'). Defaults to 'freq'.
    Returns:
        argparse.ArgumentParser: Configured argument parser for species retrieval.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[species_list_args()],
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="Path to output file or folder. If this is a folder, file will be named 'species_list.txt'.",
    )

    parser.add_argument(
        "--sortby",
        default="freq",
        choices=["freq", "alpha"],
        help="Sort species by occurrence frequency or alphabetically. Values in ['freq', 'alpha'].",
    )

    return parser


def train_parser():
    """
    Creates an argument parser for training a custom classifier with BirdNET.
    The parser includes arguments for various training parameters such as input data path, crop mode,
    output path, number of epochs, batch size, validation split ratio, learning rate, hidden units,
    dropout rate, mixup, upsampling ratio and mode, model format, model save mode, cache mode and file,
    and hyperparameter tuning options.
    Returns:
        argparse.ArgumentParser: Configured argument parser for training a custom classifier.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[
            bandpass_args(),
            audio_speed_args(),
            threads_args(),
            bs_args(cfg.TRAIN_BATCH_SIZE),
            overlap_args(help_string="Overlap of training data segments in seconds if crop_mode is 'segments'."),
        ],
    )
    c = (
        "checkpoints/custom/Custom_Classifier"
        if os.environ.get("IS_GITHUB_RUNNER", "false").lower() == "true"
        else os.path.join(SCRIPT_DIR, "checkpoints/custom/Custom_Classifier")
    )
    parser.add_argument(
        "audio_input",
        metavar="INPUT",
        help="Path to training data folder. Subfolder names are used as labels.",
    )
    parser.add_argument("--test_data", help="Path to test data folder. If not specified, a random validation split will be used.")
    parser.add_argument(
        "--crop_mode",
        default=cfg.SAMPLE_CROP_MODE,
        choices=["center", "first", "segments", "smart"],
        help="Crop mode for training data. Can be 'center', 'first', 'segments' or 'smart'.",
    )
    parser.add_argument("-o", "--output", default=c, help="Path to trained classifier model output.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=cfg.TRAIN_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=cfg.TRAIN_VAL_SPLIT,
        help="Validation split ratio. Will be ignored if test_data is set.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=cfg.TRAIN_LEARNING_RATE,
        help="Learning rate.",
    )
    parser.add_argument(
        "--focal-loss",
        dest="use_focal_loss",
        action="store_true",
        help="Use focal loss for training (helps with imbalanced classes and hard examples).",
    )
    parser.add_argument(
        "--focal-loss-gamma",
        default=cfg.FOCAL_LOSS_GAMMA,
        type=float,
        help="Focal loss gamma parameter (focusing parameter). Higher values give more weight to hard examples.",
    )
    parser.add_argument(
        "--focal-loss-alpha",
        default=cfg.FOCAL_LOSS_ALPHA,
        type=float,
        help="Focal loss alpha parameter (balancing parameter). Controls weight between positive and negative examples.",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=cfg.TRAIN_HIDDEN_UNITS,
        help="Number of hidden units. If set to >0, a two-layer classifier is used.",
    )
    parser.add_argument(
        "--dropout",
        type=lambda a: min(max(0, float(a)), 0.9),
        default=cfg.TRAIN_DROPOUT,
        help="Dropout rate. Higher values result in more regularization. Values in [0.0, 0.9].",
    )
    parser.add_argument("--label_smoothing", action="store_true", help="Whether to use label smoothing for training.")
    parser.add_argument("--mixup", action="store_true", help="Whether to use mixup for training.")
    parser.add_argument(
        "--upsampling_ratio",
        type=lambda a: min(max(0, float(a)), 1),
        default=cfg.UPSAMPLING_RATIO,
        help="Balance train data and upsample minority classes. Values between 0 and 1.",
    )
    parser.add_argument(
        "--upsampling_mode",
        default=cfg.UPSAMPLING_MODE,
        choices=["repeat", "linear", "mean", "smote"],
        help="Upsampling mode.",
    )
    parser.add_argument(
        "--model_format",
        default=cfg.TRAINED_MODEL_OUTPUT_FORMAT,
        choices=["tflite", "raven", "both"],
        help="Model output format.",
    )
    parser.add_argument(
        "--model_save_mode",
        default=cfg.TRAINED_MODEL_SAVE_MODE,
        choices=["replace", "append"],
        help="Model save mode. 'replace' will overwrite the original classification layer and 'append' will combine the original classification layer with the new one.",
    )
    parser.add_argument("--cache_mode", choices=["load", "save"], help="Cache mode. Can be 'load' or 'save'.")
    parser.add_argument("--cache_file", default=cfg.TRAIN_CACHE_FILE, help="Path to cache file.")
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).",
    )
    parser.add_argument(
        "--autotune_trials",
        type=int,
        default=cfg.AUTOTUNE_TRIALS,
        help="Number of training runs for hyperparameter tuning.",
    )
    parser.add_argument(
        "--autotune_executions_per_trial",
        type=int,
        default=cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL,
        help="The number of times a training run with a set of hyperparameters is repeated during hyperparameter tuning (this reduces the variance).",
    )

    return parser
