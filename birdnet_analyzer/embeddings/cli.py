from birdnet_analyzer import embeddings
from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    from birdnet_analyzer import cli

    parser = cli.embeddings_parser()
    args = parser.parse_args()

    embeddings(**vars(args))
