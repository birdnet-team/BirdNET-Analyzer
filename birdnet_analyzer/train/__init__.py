import birdnet_analyzer.utils as utils


@utils.runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli
    import birdnet_analyzer.config as cfg

    # Parse arguments
    parser = cli.train_parser()

    args = parser.parse_args()

    from birdnet_analyzer.train.utils import train_model  # noqa: E402

    # Config
    cfg.TRAIN_DATA_PATH = args.input
    cfg.TEST_DATA_PATH = args.test_data
    cfg.SAMPLE_CROP_MODE = args.crop_mode
    cfg.SIG_OVERLAP = args.overlap
    cfg.CUSTOM_CLASSIFIER = args.output
    cfg.TRAIN_EPOCHS = args.epochs
    cfg.TRAIN_BATCH_SIZE = args.batch_size
    cfg.TRAIN_VAL_SPLIT = args.val_split
    cfg.TRAIN_LEARNING_RATE = args.learning_rate
    cfg.TRAIN_HIDDEN_UNITS = args.hidden_units
    cfg.TRAIN_DROPOUT = args.dropout
    cfg.TRAIN_WITH_MIXUP = args.mixup if args.mixup is not None else cfg.TRAIN_WITH_MIXUP
    cfg.UPSAMPLING_RATIO = args.upsampling_ratio
    cfg.UPSAMPLING_MODE = args.upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = args.model_format
    cfg.TRAINED_MODEL_SAVE_MODE = args.model_save_mode
    cfg.TRAIN_CACHE_MODE = args.cache_mode
    cfg.TRAIN_CACHE_FILE = args.cache_file
    cfg.TFLITE_THREADS = 1
    cfg.CPU_THREADS = args.threads

    cfg.BANDPASS_FMIN = args.fmin
    cfg.BANDPASS_FMAX = args.fmax

    cfg.AUDIO_SPEED = args.audio_speed

    cfg.AUTOTUNE = args.autotune
    cfg.AUTOTUNE_TRIALS = args.autotune_trials
    cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL = args.autotune_executions_per_trial

    # Train model
    train_model()
