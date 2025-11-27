import birdnet
from birdnet.globals import ACOUSTIC_MODEL_VERSIONS, MODEL_LANGUAGES


def run_interference(
    path,
    model="birdnet",
    version: ACOUSTIC_MODEL_VERSIONS = "2.4",
    top_k: int | None = 5,
    batch_size=1,
    prefetch_ratio=2,
    overlap_duration_s=0.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    apply_sigmoid=True,
    sigmoid_sensitivity=1.0,
    speed=1.0,
    min_confidence=0.1,
    custom_species_list=None,
    label_language: MODEL_LANGUAGES = "en_us",
    classifier: str | None = None,
    cc_species_list: str | None = None,
):
    if classifier:
        if not cc_species_list:
            cc_species_list = classifier.replace(".tflite", "_Labels.txt", 1)

        model = birdnet.load_custom("acoustic", version, "tf", model=classifier, species_list=cc_species_list)
    else:
        model = birdnet.load("acoustic", version, "tf", lang=label_language)

    return model.predict(
        path,
        top_k=top_k,
        batch_size=batch_size,
        prefetch_ratio=prefetch_ratio,
        overlap_duration_s=overlap_duration_s,
        bandpass_fmin=bandpass_fmin,
        bandpass_fmax=bandpass_fmax,
        apply_sigmoid=apply_sigmoid,
        sigmoid_sensitivity=sigmoid_sensitivity,
        speed=speed,
        default_confidence_threshold=min_confidence,
        custom_species_list=custom_species_list,
    )
