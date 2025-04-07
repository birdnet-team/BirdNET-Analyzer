from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli
    from birdnet_analyzer import train

    # Parse arguments
    parser = cli.train_parser()

    args = parser.parse_args()

    train(**vars(args))
