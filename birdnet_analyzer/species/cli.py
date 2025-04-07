from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli
    from birdnet_analyzer import species

    # Parse arguments
    parser = cli.species_parser()

    args = parser.parse_args()

    species(**vars(args))
