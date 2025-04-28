from typing import Literal


def train(
    audio_input: str,
    output: str = "checkpoints/custom/Custom_Classifier",
    test_data: str | None = None,
    *,
    crop_mode: Literal["center", "first", "segments"] = "center",
    overlap: float = 0.0,
    epochs: int = 50,
    batch_size: int = 32,
    val_split: float = 0.2,
    learning_rate: float = 0.0001,
    use_focal_loss: bool = False,
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: float = 0.25,
    hidden_units: int = 0,
    dropout: float = 0.0,
    label_smoothing: bool = False,
    mixup: bool = False,
    upsampling_ratio: float = 0.0,
    upsampling_mode: Literal["repeat", "mean", "smote"] = "repeat",
    model_format: Literal["tflite", "raven", "both"] = "tflite",
    model_save_mode: Literal["replace", "append"] = "replace",
    cache_mode: Literal["load", "save"] | None = None,
    cache_file: str = "train_cache.npz",
    threads: int = 1,
    fmin: float = 0.0,
    fmax: float = 15000.0,
    audio_speed: float = 1.0,
    autotune: bool = False,
    autotune_trials: int = 50,
    autotune_executions_per_trial: int = 1,
):
    """
    Trains a custom classifier model using the BirdNET-Analyzer framework.
    Args:
        audio_input (str): Path to the training data directory.
        test_data (str, optional): Path to the test data directory. Defaults to None. If not specified, a validation split will be used.
        output (str, optional): Path to save the trained model. Defaults to "checkpoints/custom/Custom_Classifier".
        crop_mode (Literal["center", "first", "segments", "smart"], optional): Mode for cropping audio samples. Defaults to "center".
        overlap (float, optional): Overlap ratio for audio segments. Defaults to 0.0.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        val_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
        use_focal_loss (bool, optional): Whether to use focal loss for training. Defaults to False.
        focal_loss_gamma (float, optional): Gamma parameter for focal loss. Defaults to 2.0.
        focal_loss_alpha (float, optional): Alpha parameter for focal loss. Defaults to 0.25.
        hidden_units (int, optional): Number of hidden units in the model. Defaults to 0.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
        label_smoothing (bool, optional): Whether to use label smoothing. Defaults to False.
        mixup (bool, optional): Whether to use mixup data augmentation. Defaults to False.
        upsampling_ratio (float, optional): Ratio for upsampling underrepresented classes. Defaults to 0.0.
        upsampling_mode (Literal["repeat", "mean", "smote"], optional): Mode for upsampling. Defaults to "repeat".
        model_format (Literal["tflite", "raven", "both"], optional): Format to save the trained model. Defaults to "tflite".
        model_save_mode (Literal["replace", "append"], optional): Save mode for the model. Defaults to "replace".
        cache_mode (Literal["load", "save"] | None, optional): Cache mode for training data. Defaults to None.
        cache_file (str, optional): Path to the cache file. Defaults to "train_cache.npz".
        threads (int, optional): Number of CPU threads to use. Defaults to 1.
        fmin (float, optional): Minimum frequency for bandpass filtering. Defaults to 0.0.
        fmax (float, optional): Maximum frequency for bandpass filtering. Defaults to 15000.0.
        audio_speed (float, optional): Speed factor for audio playback. Defaults to 1.0.
        autotune (bool, optional): Whether to use hyperparameter autotuning. Defaults to False.
        autotune_trials (int, optional): Number of trials for autotuning. Defaults to 50.
        autotune_executions_per_trial (int, optional): Number of executions per autotuning trial. Defaults to 1.
    Returns:
        None
    """
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer.train.utils import train_model
    from birdnet_analyzer.utils import ensure_model_exists

    ensure_model_exists()

    # Config
    cfg.TRAIN_DATA_PATH = audio_input
    cfg.TEST_DATA_PATH = test_data
    cfg.SAMPLE_CROP_MODE = crop_mode
    cfg.SIG_OVERLAP = overlap
    cfg.CUSTOM_CLASSIFIER = output
    cfg.TRAIN_EPOCHS = epochs
    cfg.TRAIN_BATCH_SIZE = batch_size
    cfg.TRAIN_VAL_SPLIT = val_split
    cfg.TRAIN_LEARNING_RATE = learning_rate
    cfg.TRAIN_WITH_FOCAL_LOSS = use_focal_loss if use_focal_loss is not None else cfg.TRAIN_WITH_FOCAL_LOSS
    cfg.FOCAL_LOSS_GAMMA = focal_loss_gamma
    cfg.FOCAL_LOSS_ALPHA = focal_loss_alpha
    cfg.TRAIN_HIDDEN_UNITS = hidden_units
    cfg.TRAIN_DROPOUT = dropout
    cfg.TRAIN_WITH_LABEL_SMOOTHING = label_smoothing if label_smoothing is not None else cfg.TRAIN_WITH_LABEL_SMOOTHING
    cfg.TRAIN_WITH_MIXUP = mixup if mixup is not None else cfg.TRAIN_WITH_MIXUP
    cfg.UPSAMPLING_RATIO = upsampling_ratio
    cfg.UPSAMPLING_MODE = upsampling_mode
    cfg.TRAINED_MODEL_OUTPUT_FORMAT = model_format
    cfg.TRAINED_MODEL_SAVE_MODE = model_save_mode
    cfg.TRAIN_CACHE_MODE = cache_mode
    cfg.TRAIN_CACHE_FILE = cache_file
    cfg.TFLITE_THREADS = 1
    cfg.CPU_THREADS = threads

    cfg.BANDPASS_FMIN = fmin
    cfg.BANDPASS_FMAX = fmax

    cfg.AUDIO_SPEED = audio_speed

    cfg.AUTOTUNE = autotune
    cfg.AUTOTUNE_TRIALS = autotune_trials
    cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL = autotune_executions_per_trial

    # Train model
    train_model()
