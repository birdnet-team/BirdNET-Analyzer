from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    from birdnet_analyzer import cli, train

    # Parse arguments
    parser = cli.train_parser()

    args = parser.parse_args()

    train(**vars(args))
