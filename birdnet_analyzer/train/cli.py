from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    from birdnet_analyzer import cli, params, train

    parser = cli.train_parser()
    cli.apply_params_file_defaults(parser, params.load_train_params)
    args = parser.parse_args()

    train_args = vars(args)
    train_args.pop("load_params")  # already applied as defaults

    train(**train_args)
