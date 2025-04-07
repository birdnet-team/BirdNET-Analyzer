import birdnet_analyzer.utils as utils


@utils.runtime_error_handler
def main():
    import birdnet_analyzer.cli as cli
    from birdnet_analyzer import search

    parser = cli.search_parser()
    args = parser.parse_args()

    search(**vars(args))
