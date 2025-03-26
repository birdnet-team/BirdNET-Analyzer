import birdnet_analyzer.utils as utils


@utils.runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli

    parser = cli.embeddings_parser()
    args = parser.parse_args()

    from birdnet_analyzer.embeddings.utils import run  # noqa: E402

    run(args.input, args.database, args.overlap, args.audio_speed, args.fmin, args.fmax, args.threads, args.batchsize)
