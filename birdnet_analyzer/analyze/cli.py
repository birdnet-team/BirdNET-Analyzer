import contextlib

from birdnet_analyzer import analyze
from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    import os
    from multiprocessing import freeze_support

    from birdnet_analyzer import cli, params

    freeze_support()

    parser = cli.analyzer_parser()
    cli.apply_params_file_defaults(parser, params.load_analysis_params)
    args = parser.parse_args()

    with contextlib.suppress(Exception):
        if os.get_terminal_size().columns >= 64:
            print(cli.ASCII_LOGO, flush=True)

    if (
        "additional_columns" in args
        and args.additional_columns
        and ("csv" not in args.rtype and "parquet" not in args.rtype)
    ):
        import warnings

        warnings.warn(
            "The --additional_columns argument is only valid for CSV output."
            "It will be ignored.",
            stacklevel=1,
        )

    if args.use_perch and args.classifier:
        raise ValueError(
            "The --use_perch and --classifier arguments cannot be used together."
        )

    analyze_args = vars(args)
    analyze_args.pop("use_perch")  # handled via model param
    analyze_args.pop("load_params")  # already applied as defaults

    analyze(**analyze_args)
