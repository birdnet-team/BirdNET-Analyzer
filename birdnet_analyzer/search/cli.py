from birdnet_analyzer import utils


@utils.runtime_error_handler
def main():
    from birdnet_analyzer import cli, search

    parser = cli.search_parser()
    args = parser.parse_args()

    search(**vars(args))
