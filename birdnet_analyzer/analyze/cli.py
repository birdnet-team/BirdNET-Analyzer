from birdnet_analyzer.utils import runtime_error_handler
from birdnet_analyzer import analyze


@runtime_error_handler
def main():
    import os
    from multiprocessing import freeze_support

    import birdnet_analyzer.cli as cli

    # Freeze support for executable
    freeze_support()

    parser = cli.analyzer_parser()

    args = parser.parse_args()

    try:
        if os.get_terminal_size().columns >= 64:
            print(cli.ASCII_LOGO, flush=True)
    except Exception:
        pass

    analyze(**vars(args))
