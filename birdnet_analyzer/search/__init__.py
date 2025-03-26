import birdnet_analyzer.utils as utils


@utils.runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli

    parser = cli.search_parser()
    args = parser.parse_args()

    import os
    from birdnet_analyzer.search.utils import get_database, get_search_results
    import birdnet_analyzer.audio as audio
    import birdnet_analyzer.config as cfg

    # Create output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the database
    db = get_database(args.database)

    try:
        settings = db.get_metadata("birdnet_analyzer_settings")
    except:
        raise ValueError("No settings present in database.")

    fmin = settings["BANDPASS_FMIN"]
    fmax = settings["BANDPASS_FMAX"]
    audio_speed = settings["AUDIO_SPEED"]

    # Execute the search
    results = get_search_results(
        args.queryfile, db, args.n_results, audio_speed, fmin, fmax, args.score_function, args.crop_mode, args.overlap
    )

    # Save the results
    for i, r in enumerate(results):
        embedding_source = db.get_embedding_source(r.embedding_id)
        file = embedding_source.source_id
        offset = embedding_source.offsets[0] * audio_speed
        duration = cfg.SIG_LENGTH * audio_speed
        sig, rate = audio.open_audio_file(file, offset=offset, duration=duration, sample_rate=None)
        result_path = os.path.join(args.output, f"search_result_{i + 1}_score_{r.sort_score:.5f}.wav")
        audio.save_signal(sig, result_path, rate)
