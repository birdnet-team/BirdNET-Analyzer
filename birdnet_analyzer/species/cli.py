from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    from birdnet_analyzer import cli, species

    # Parse arguments
    parser = cli.species_parser()

    args = parser.parse_args()

    species(**vars(args))
