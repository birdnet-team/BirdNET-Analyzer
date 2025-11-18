import birdnet_analyzer.config as cfg
import birdnet


def run_interference(input_data):
    model = birdnet.load("acoustic", "2.4", "tf", lang=cfg.TRANSLATED_LABELS)
    results = model.predict(
        input_data,
        top_k=cfg.TOP_N,
        apply_sigmoid=cfg.APPLY_SIGMOID,
        bandpass_fmax=cfg.BANDPASS_FMAX,
        bandpass_fmin=cfg.BANDPASS_FMIN,
        overlap_duration_s=cfg.SIG_OVERLAP,
        batch_size=cfg.BATCH_SIZE,
        default_confidence_threshold=cfg.MIN_CONFIDENCE,
        custom_species_list=cfg.SPECIES_LIST_FILE,
        sigmoid_sensitivity=cfg.SIGMOID_SENSITIVITY,
        prefetch_ratio=2
    )

    return results
