from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from birdnet_analyzer.config import (
        SAMPLE_CROP_MODES,
        TRAINED_MODEL_OUTPUT_FORMATS,
        TRAINED_MODEL_SAVE_MODES,
        UPSAMPLING_MODES,
    )


def train(
    audio_input: str,
    output: str = "checkpoints/custom/Custom_Classifier",
    test_data: str | None = None,
    *,
    crop_mode: SAMPLE_CROP_MODES = "center",
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
    upsampling_mode: UPSAMPLING_MODES = "repeat",
    model_format: TRAINED_MODEL_OUTPUT_FORMATS = "tflite",
    model_save_mode: TRAINED_MODEL_SAVE_MODES = "replace",
    save_cache_to: str | None = None,
    threads: int = 1,
    fmin: float = 0.0,
    fmax: float = 15000.0,
    audio_speed: float = 1.0,
    autotune: bool = False,
    autotune_trials: int = 50,
    autotune_n_repeats: int = 1,
    autotune_n_splits: int = 3,
):
    """
    Trains a custom classifier model using the BirdNET-Analyzer framework.
    Args:
        audio_input (str): Path to the training data directory or path to a cache file
                           ("train_cache.npz" for example).
        test_data (str, optional): Path to the test data directory. Defaults to None.
                                   If not specified, a validation split will be used.
        output (str, optional): Path to save the trained model.
                                Defaults to "checkpoints/custom/Custom_Classifier".
        crop_mode (Literal["center", "first", "segments", "smart"], optional): Mode for
            cropping audio samples. Defaults to "center".
        overlap (float, optional): Overlap ratio for audio segments. Defaults to 0.0.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        val_split (float, optional): Fraction of data to use for validation.
                                     Defaults to 0.2.
        learning_rate (float, optional): Learning rate for the optimizer.
                                         Defaults to 0.0001.
        use_focal_loss (bool, optional): Whether to use focal loss for training.
                                         Defaults to False.
        focal_loss_gamma (float, optional): Gamma parameter for focal loss.
                                            Defaults to 2.0.
        focal_loss_alpha (float, optional): Alpha parameter for focal loss.
                                            Defaults to 0.25.
        hidden_units (int, optional): Number of hidden units in the model. Defaults to 0
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
        label_smoothing (bool, optional): Whether to use label smoothing.
                                          Defaults to False.
        mixup (bool, optional): Whether to use mixup data augmentation.
                                Defaults to False.
        upsampling_ratio (float, optional): Ratio for upsampling underrepresented
                                            classes. Defaults to 0.0.
        upsampling_mode (Literal["repeat", "mean", "smote"], optional): Mode for
            upsampling. Defaults to "repeat".
        model_format (Literal["tflite", "raven", "both"], optional): Format to save the
            trained model. Defaults to "tflite".
        model_save_mode (Literal["replace", "append"], optional): Save mode for the
            model. Defaults to "replace".
        save_cache_to (str | None, optional): Path to save the cache file.
                                              Defaults to None.
        threads (int, optional): Number of CPU threads to use. Defaults to 1.
        fmin (float, optional): Minimum frequency for bandpass filtering.
                                Defaults to 0.0.
        fmax (float, optional): Maximum frequency for bandpass filtering.
                                Defaults to 15000.0.
        audio_speed (float, optional): Speed factor for audio playback. Defaults to 1.0.
        autotune (bool, optional): Whether to use hyperparameter autotuning.
                                   Defaults to False.
        autotune_trials (int, optional): Number of trials for autotuning.
                                         Defaults to 50.
        autotune_n_repeats (int, optional): Number of times to repeat each trial during
                                           hyperparameter tuning. Defaults to 1.
        autotune_n_splits (int, optional): Number of splits for cross-validation during
                                          hyperparameter tuning. Defaults to 3.
    Returns:
        None
    """
    from birdnet_analyzer.train.utils import train_model

    train_model(
        audio_input,
        output=output,
        test_data=test_data,
        crop_mode=crop_mode,
        overlap=overlap,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        learning_rate=learning_rate,
        use_focal_loss=use_focal_loss,
        focal_loss_gamma=focal_loss_gamma,
        focal_loss_alpha=focal_loss_alpha,
        hidden_units=hidden_units,
        dropout=dropout,
        label_smoothing=label_smoothing,
        mixup=mixup,
        upsampling_ratio=upsampling_ratio,
        upsampling_mode=upsampling_mode,
        model_format=model_format,
        model_save_mode=model_save_mode,
        save_cache_to=save_cache_to,
        threads=threads,
        fmin=fmin,
        fmax=fmax,
        audio_speed=audio_speed,
        autotune=autotune,
        autotune_trials=autotune_trials,
        autotune_n_repeats=autotune_n_repeats,
        autotune_n_splits=autotune_n_splits,
    )
