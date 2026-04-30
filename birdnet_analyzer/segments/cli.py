from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    from birdnet_analyzer import cli, segments

    parser = cli.segments_parser()
    args = parser.parse_args()

    segments(**vars(args))
