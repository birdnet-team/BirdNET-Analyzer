# ruff: noqa: E501
import argparse
import logging
import os
from typing import cast, get_args

from birdnet.globals import (
    ACOUSTIC_MODEL_VERSIONS,
    MODEL_LANGUAGE_EN_US,
)

from birdnet_analyzer.config import (
    ALL_MODEL_LANGUAGES,
    AUTOTUNE_METRICS,
    DEFAULT_ACOUSTIC_MODEL_VERSION,
    GEO_MODEL_LANGUAGES,
    TRAINED_MODEL_OUTPUT_FORMATS,
)
from birdnet_analyzer.logs import setup_logging

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


def apply_params_file_defaults(parser, loader, argv=None):
    """Makes the values of a ``--load_params`` file the defaults of a parser.

    Reads the file before the actual parsing, so arguments given on the command line
    override the values from the file, which in turn override the built-in defaults.
    Values the parser has no argument for are ignored.

    Args:
        parser: The fully built argument parser.
        loader: Reads the parameters file into keyword arguments, e.g.
            :func:`birdnet_analyzer.params.load_analysis_params`.
        argv: The command line to read the file path from. Defaults to ``sys.argv``.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--load_params")
    known, _ = pre_parser.parse_known_args(argv)

    if not known.load_params:
        return

    try:
        values = loader(known.load_params)
    except ValueError as e:
        parser.error(str(e))

    dests = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in values.items() if key in dests})


def load_params_args(run: str, files_hint: str):
    """
    Creates an argument parser for reading the settings of a previous run.

    Args:
        run: What the parameters file belongs to, e.g. "analysis".
        files_hint: The file names to point the user to.

    Returns:
        argparse.ArgumentParser: The argument parser with the `--load_params`
        argument.
    """
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--load_params",
        metavar="PARAMS_FILE",
        help=f"Read default settings from the parameters file of a previous {run} "
        f"({files_hint}). Arguments given on the command line take precedence. "
        "Parameters files of earlier BirdNET-Analyzer versions are understood too.",
    )

    return p


def store_model_action(model_name: str):
    class StoreModelAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=False,
            required=False,
            help=None,  # noqa: A002
        ):
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=0,
                const=True,
                default=default,
                required=required,
                help=help,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, True)
            namespace.model = model_name

    return StoreModelAction


def set_model_action(model_name: str):
    class SetModelAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)

            namespace.model = model_name

    return SetModelAction


def birdnet_arg():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--birdnet",
        default=DEFAULT_ACOUSTIC_MODEL_VERSION,
        const=DEFAULT_ACOUSTIC_MODEL_VERSION,
        nargs="?",
        choices=get_args(ACOUSTIC_MODEL_VERSIONS),
        action=set_model_action("birdnet"),
        help="Use the BirdNET model. Specify the version to use.",
    )

    return p


class _VerbosityAction(argparse.Action):
    """Reconfigures the console log level as soon as the flag is parsed.

    Stores nothing in the namespace, so the flag never reaches the keyword arguments
    the entry points pass into the feature functions.
    """

    def __init__(self, option_strings, dest, level=logging.INFO, **kwargs):
        super().__init__(
            option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs
        )
        self._level = level

    def __call__(self, parser, namespace, values, option_string=None):
        setup_logging(self._level)


def verbosity_args():
    """
    Creates an argument parser for the console verbosity.

    -q is not available as a short form of --quiet, because search uses it for the
    query file.

    Returns:
        argparse.ArgumentParser: The argument parser with the verbosity arguments.
    """
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-v",
        "--verbose",
        action=_VerbosityAction,
        level=logging.DEBUG,
        help="Also show debug messages and stacktraces on the console.",
    )
    g.add_argument(
        "--quiet",
        action=_VerbosityAction,
        level=logging.WARNING,
        help="Only show warnings and errors on the console.",
    )

    return p


def io_args():
    """Argument parser for input and output paths."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "audio_input",
        metavar="INPUT",
        help="Path to input file or folder.",
    )
    p.add_argument(
        "-o", "--output", help="Path to output folder. Defaults to the input path."
    )

    return p


def bandpass_args():
    """Argument parser for bandpass filter frequencies (--fmin, --fmax)."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--fmin",
        type=lambda a: max(0, min(15000, int(a))),
        default=0,
        help="Minimum frequency for bandpass filter in Hz.",
    )
    p.add_argument(
        "--fmax",
        type=lambda a: max(0, min(15000, int(a))),
        default=15000,
        help="Maximum frequency for bandpass filter in Hz.",
    )

    return p


def species_list_args(add_species_list_hint=False):
    """Argument parser for the species-list filter arguments."""
    p = argparse.ArgumentParser(add_help=False)

    slist_hint = (
        " Cannot be used together with --slist, use either the species list or the location coordinates."
        if add_species_list_hint
        else ""
    )

    p.add_argument(
        "--lat",
        type=float,
        help="Recording location latitude in decimal degrees." + slist_hint,
    )
    p.add_argument(
        "--lon",
        type=float,
        help="Recording location longitude in decimal degrees." + slist_hint,
    )
    p.add_argument(
        "--week",
        type=int,
        help="Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Only effective when --lat and --lon are provided. Leave blank for year-round species list.",
    )
    p.add_argument(
        "--sf_thresh",
        type=lambda a: max(0.0001, min(0.99, float(a))),
        default=0.03,
        help="Minimum species occurrence frequency threshold for location filter. Values in [0.0001, 0.99].",
    )

    return p


def species_args():
    """Argument parser for species arguments, including the species-list arguments."""
    p = species_list_args(add_species_list_hint=True)

    p.add_argument(
        "--slist",
        help='Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". Cannot be used together with --lat and --lon, use either the species list or the location coordinates.',
    )

    return p


def sigmoid_args():
    """Argument parser for the sigmoid detection sensitivity."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--sensitivity",
        type=lambda a: min(1.5, max(0.5, float(a))),
        default=1.0,
        help="Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Values other than 1.0 will shift the sigmoid function on the x-axis. Use complementary to the cut-off threshold. Only applies to BirdNET 2.4 and custom classifiers; ignored for BirdNET 3.0 (which applies the sigmoid inside the model) and Perch.",
    )

    return p


def overlap_args(help_string="Overlap of prediction segments. Values in [0.0, 2.9]."):
    """Argument parser for the overlap of prediction segments."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--overlap",
        type=lambda a: max(0.0, min(4.9, float(a))),
        default=0.0,
        help=help_string,
    )

    return p


def audio_speed_args():
    """Argument parser for the audio speed factor (--audio_speed)."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--audio_speed",
        type=lambda a: max(0.01, float(a)),
        default=1.0,
        help="Speed factor for audio playback. Values < 1.0 will slow down the audio, values > 1.0 will speed it up. At a 10x decrease (audio speed 0.1), a 384 kHz recording becomes a 38.4 kHz recording.",
    )

    return p


def threads_args():
    """Argument parser for --threads (default: half the CPU cores, capped at 8)."""
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
    """Argument parser for the minimum confidence threshold (--min_conf)."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--min_conf",
        default=0.25,
        type=lambda a: max(0.00001, min(0.99, float(a))),
        help="Minimum confidence threshold. Values in [0.00001, 0.99].",
    )

    return p


def locale_args(languages=None):
    """Argument parser for the --locale of translated species common names.

    Args:
        languages: The locale codes to offer. Defaults to the union of every model
            version's languages; the birdnet library validates the concrete
            (model version, locale) pair when the model is loaded. Pass a narrower
            set for commands bound to a single model (e.g. the geo model).
    """
    p = argparse.ArgumentParser(add_help=False)

    locale_choices = list(languages) if languages is not None else ALL_MODEL_LANGUAGES
    p.add_argument(
        "-l",
        "--locale",
        default=cast("str", MODEL_LANGUAGE_EN_US),
        choices=locale_choices,
        help="Locale for translated species common names.",
    )

    return p


def bs_args(default=1):
    """Argument parser for the batch size (-b/--batch_size)."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "-b",
        "--batch_size",
        type=lambda a: max(1, int(a)),
        default=default,
        help="Number of samples to process at the same time.",
    )

    return p


def computing_resources_args():
    """Argument parser for worker and producer process counts."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "--n_workers",
        type=int,
        help="Number of worker processes for audio processing. Defaults to number of CPU cores.",
    )
    p.add_argument(
        "--n_producers",
        type=int,
        default=1,
        help="Number of producer processes for audio processing. Defaults to 1.",
    )

    return p


def db_args():
    """Argument parser for the database path (-db/--database)."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument(
        "-db",
        "--database",
        help="Path to the database folder.",
        required=True,
    )

    return p


def analyzer_parser():
    """Build the argument parser for the analyze CLI."""
    from birdnet_analyzer.analyze import POSSIBLE_ADDITIONAL_COLUMNS

    parents = [
        birdnet_arg(),
        io_args(),
        bandpass_args(),
        species_args(),
        sigmoid_args(),
        overlap_args(),
        audio_speed_args(),
        min_conf_args(),
        locale_args(),
        bs_args(),
        computing_resources_args(),
        load_params_args("analysis", "birdnet.analyze-params.csv"),
        verbosity_args(),
    ]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
    )

    class UniqueSetAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, {v.lower() for v in values})

    parser.add_argument(
        "--rtype",
        default={"table"},
        choices=["table", "audacity", "kaleidoscope", "csv", "parquet"],
        nargs="+",
        help="Specifies output format. Values in `['table', 'audacity',  'kaleidoscope', 'csv', 'parquet']`.",
        action=UniqueSetAction,
    )
    parser.add_argument(
        "--additional_columns",
        choices=POSSIBLE_ADDITIONAL_COLUMNS,
        nargs="+",
        help="Additional columns to include in the output, only applied to the csv and parquet output formats.",
        action=UniqueSetAction,
    )
    parser.add_argument(
        "-c",
        "--classifier",
        help="Path to custom trained classifier. If set, --lat, --lon and --locale are ignored.",
    )
    parser.add_argument(
        "--cc_species_list",
        help="Path to custom species list file for the custom classifier. The default search path is <custom_classifier_without_extension>_Labels.txt in the same directory.",
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
        help="Maximum number of consecutive detections above the threshold to merge for each detected species. This will result in fewer entries in the result file with segments longer than 3 seconds. Set to 0 or 1 to disable merging. We use the mean of the top 3 scores from all consecutive detections for merging.",
    )
    parser.add_argument(
        "--use_perch",
        action=store_model_action("perch"),
        help="Use the Perch model for detection.",
    )
    parser.add_argument(
        "--split_tables",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Saves separate result tables for each input audio file in the output.",
    )
    parser.add_argument(
        "--strict",
        dest="strict_species_list",
        action="store_true",
        help="Fail if any species in --slist is not in the model. By default such "
        "species are skipped with a warning: labels drift across model versions (a "
        "renamed genus or common name), so the list is matched to the model by "
        "scientific and common name first, and only genuinely unknown species are "
        "dropped.",
    )
    parser.set_defaults(model="birdnet")

    return parser


def embeddings_parser():
    """Build the argument parser for extracting feature embeddings."""

    parents = [
        db_args(),
        bandpass_args(),
        audio_speed_args(),
        overlap_args(),
        bs_args(default=8),
        computing_resources_args(),
        verbosity_args(),
    ]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="audio_input",
        help="Path to input file or folder, relative to the audio root.",
    )

    parser.add_argument(
        "--file_output",
        help="Saves all embeddings contained in the database in a csv file.",
    )

    return parser


def search_parser():
    """Build the argument parser for searching BirdNET embeddings."""

    parents = [overlap_args(), db_args(), verbosity_args()]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=parents
    )

    parser.add_argument("-q", "--queryfile", help="Path to the query file.")
    parser.add_argument("-o", "--output", help="Path to the output folder.")
    parser.add_argument(
        "--n_results", default=10, type=int, help="Number of results to return."
    )

    # TODO: use choice argument.
    parser.add_argument(
        "--score_function",
        default="cosine",
        choices=["cosine", "euclidean", "dot"],
        help="Scoring function to use. Choose 'cosine', 'euclidean' or 'dot'. Defaults to 'cosine'.",
    )
    parser.add_argument(
        "--crop_mode",
        default="center",
        choices=["center", "first", "segments"],
        help="Crop mode for the query sample. Can be 'center', 'first' or 'segments'.",
    )

    parser.add_argument(
        "--audio_root",
        help="Path to the root directory of audio files.",
    )

    return parser


def segments_parser():
    """Build the argument parser for extracting segments from audio files."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[
            audio_speed_args(),
            threads_args(),
            min_conf_args(),
            verbosity_args(),
        ],
    )

    parser.add_argument(
        "audio_input", metavar="INPUT", help="Path to folder containing audio files."
    )
    parser.add_argument(
        "-r",
        "--results",
        help="Path to folder containing result files. Defaults to the `input` path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output folder path for extracted segments. Defaults to the `input` path.",
    )
    parser.add_argument(
        "--max_segments",
        type=lambda a: max(1, int(a)),
        default=100,
        help="Number of randomly extracted segments per species.",
    )
    parser.add_argument(
        "--seg_length",
        type=lambda a: max(1.0, float(a)),
        default=3.0,
        help="Minimum length of extracted segments in seconds. If a segment is shorter than this value, it will be padded with audio from the source file.",
    )
    parser.add_argument(
        "--max_conf",
        default=1.0,
        type=lambda a: max(0.00001, min(1.0, float(a))),
        help="Maximum confidence threshold. Values in [0.00001, 1.0].",
    )
    parser.add_argument(
        "--collection_mode",
        default="random",
        choices=["random", "confidence", "balanced"],
        help="Collection mode for selecting the segments. Can be 'random' or 'confidence'.",
    )
    parser.add_argument(
        "--n_bins",
        type=lambda a: max(2, int(a)),
        default=10,
        help="Number of bins to use for the balanced collection mode",
    )

    return parser


def species_parser():
    """Build the argument parser for retrieving a species list for a location."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[
            species_list_args(),
            # The species list comes from the geo model, so only its languages apply.
            locale_args(languages=GEO_MODEL_LANGUAGES),
            verbosity_args(),
        ],
    )

    parser.add_argument(
        "output",
        metavar="OUTPUT",
        help="Path to output file or folder. If this is a folder, file will be named 'species_list.txt'.",
    )

    return parser


def train_parser():
    """Build the argument parser for training a custom classifier."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[
            bandpass_args(),
            audio_speed_args(),
            threads_args(),
            bs_args(32),
            overlap_args(
                help_string="Overlap of training data segments in seconds if crop_mode is 'segments'."
            ),
            load_params_args("training run", "*.birdnet.train-params.csv"),
            verbosity_args(),
        ],
    )
    # Relative to the working directory: writing into the installed package
    # directory fails on read-only installs and loses the classifier on upgrade.
    c = os.path.join("checkpoints", "custom", "Custom_Classifier")

    parser.add_argument(
        "audio_input",
        metavar="INPUT",
        help="Path to training data folder. Subfolder names are used as labels. Can also be path to cache file",
    )
    parser.add_argument(
        "--test_data",
        help="Path to test data folder. If not specified, a random validation split will be used.",
    )
    parser.add_argument(
        "--crop_mode",
        default="center",
        choices=["center", "first", "segments", "smart"],
        help="Crop mode for training data. Can be 'center', 'first', 'segments' or 'smart'.",
    )
    parser.add_argument(
        "-o", "--output", default=c, help="Path to trained classifier model output."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="A small percentage of the training data that is not used for training, but for validation scores during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--focal-loss",
        dest="use_focal_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use focal loss for training (helps with imbalanced classes and hard examples).",
    )
    parser.add_argument(
        "--focal-loss-gamma",
        default=2.0,
        type=float,
        help="Focal loss gamma parameter (focusing parameter). Higher values give more weight to hard examples.",
    )
    parser.add_argument(
        "--focal-loss-alpha",
        default=0.25,
        type=float,
        help="Focal loss alpha parameter (balancing parameter). Controls weight between positive and negative examples.",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=0,
        help="Number of hidden units. If set to >0, a two-layer classifier is used.",
    )
    parser.add_argument(
        "--dropout",
        type=lambda a: min(max(0, float(a)), 0.9),
        default=0.0,
        help="Dropout rate. Higher values result in more regularization. Values in [0.0, 0.9].",
    )
    parser.add_argument(
        "--label_smoothing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use label smoothing for training.",
    )
    parser.add_argument(
        "--mixup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use mixup for training.",
    )
    parser.add_argument(
        "--upsampling_ratio",
        type=lambda a: min(max(0, float(a)), 1),
        default=0.0,
        help="Balance train data and upsample minority classes. Values between 0 and 1.",
    )
    parser.add_argument(
        "--upsampling_mode",
        default="repeat",
        choices=["repeat", "linear", "mean", "smote"],
        help="Upsampling mode.",
    )
    parser.add_argument(
        "--model_formats",
        nargs="+",
        default=["tflite"],
        choices=get_args(TRAINED_MODEL_OUTPUT_FORMATS),
        help="Model output format(s). One or more of 'tflite', 'raven', 'detached'.",
    )
    parser.add_argument(
        "--model_save_mode",
        default="replace",
        choices=["replace", "append"],
        help="Model save mode. 'replace' will overwrite the original classification layer and 'append' will combine the original classification layer with the new one.",
    )
    parser.add_argument(
        "--save_cache_to", default="train_cache.npz", help="Path to cache file."
    )
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use automatic hyperparameter tuning (this will execute multiple training runs to search for optimal hyperparameters).",
    )
    parser.add_argument(
        "--autotune_trials",
        type=int,
        default=50,
        help="Number of training runs for hyperparameter tuning.",
    )
    parser.add_argument(
        "--autotune_n_splits",
        type=int,
        default=1,
        help="Number of folds for cross-validation during hyperparameter tuning. If set to >1, the training data will be split into k folds and each training run will be executed k times with a different fold as validation data and the rest as training data. Keep in mind that this will significantly increase the runtime of the hyperparameter tuning process.",
    )
    parser.add_argument(
        "--autotune_n_repeats",
        type=int,
        default=1,
        help="Number of repetitions for each training run during hyperparameter tuning. If set to >1, each training run will be executed multiple times and the average validation score across all repetitions will be used as the score for the training run. This can help to get more robust estimates of the validation scores for each training run, but it will also increase the runtime of the hyperparameter tuning process.",
    )
    parser.add_argument(
        "--autotune_metric",
        default="val_AUPRC",
        choices=get_args(AUTOTUNE_METRICS),
        help="Metric to optimize during hyperparameter tuning. This can be any metric that is returned by the training process and is included in the training history object. Common choices are 'val_loss', 'val_AUPRC' or 'val_AUROC'.",
    )

    return parser
