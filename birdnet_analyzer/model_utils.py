import birdnet

import birdnet_analyzer.config as cfg


def run_interference(path):
    model = birdnet.load("acoustic", "2.4", "tf", lang=cfg.LABEL_LANGUAGE)

    predictions = model.predict(
        path,
        top_k=cfg.TOP_N,
        batch_size=cfg.BATCH_SIZE,
        prefetch_ratio=2,
        overlap_duration_s=cfg.SIG_OVERLAP,
        bandpass_fmin=cfg.BANDPASS_FMIN,
        bandpass_fmax=cfg.BANDPASS_FMAX,
        apply_sigmoid=cfg.APPLY_SIGMOID,
        sigmoid_sensitivity=cfg.SIGMOID_SENSITIVITY,
        audio_speed=cfg.AUDIO_SPEED, # TODO
        default_confidence_threshold=cfg.MIN_CONFIDENCE,
        custom_species_list=cfg.SPECIES_LIST_FILE,
    )
