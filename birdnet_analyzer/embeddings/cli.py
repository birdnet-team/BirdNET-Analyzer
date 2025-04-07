from birdnet_analyzer.utils import runtime_error_handler

from birdnet_analyzer import embeddings


@runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli

    parser = cli.embeddings_parser()
    args = parser.parse_args()

    embeddings(**vars(args))
