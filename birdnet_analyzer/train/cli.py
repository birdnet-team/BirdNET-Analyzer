import logging

from birdnet_analyzer.utils import runtime_error_handler

logger = logging.getLogger(__name__)


def train_kwargs(argv=None) -> dict:
    """Parses a training command line into keyword arguments for ``train()``.

    A learning rate read with ``--load_params`` applies only to the base model it was
    recorded for; training on another one uses that model's default instead.

    Args:
        argv: The command line arguments. Defaults to ``sys.argv``.
    """
    from birdnet_analyzer import cli, params
    from birdnet_analyzer.base_models import cache_model_version

    parser = cli.train_parser()
    recorded = cli.apply_params_file_defaults(parser, params.load_train_params, argv)
    args = parser.parse_args(argv)
    recorded_base = recorded.get("birdnet")
    trained_base = args.birdnet

    if args.audio_input.endswith(".npz"):
        trained_base = cache_model_version(args.audio_input) or trained_base

    if (
        "learning_rate" in recorded
        and recorded_base is not None
        and recorded_base != trained_base
    ):
        parser.set_defaults(learning_rate=None)
        args = parser.parse_args(argv)

        if args.learning_rate is None:
            logger.warning(
                "The learning rate in %s was recorded for BirdNET %s; training on "
                "BirdNET %s uses the default for that version instead.",
                args.load_params,
                recorded_base,
                trained_base,
            )

    kwargs = vars(args)
    kwargs.pop("load_params")  # already applied as defaults

    return kwargs


@runtime_error_handler
def main():
    from birdnet_analyzer import train

    train(**train_kwargs())
