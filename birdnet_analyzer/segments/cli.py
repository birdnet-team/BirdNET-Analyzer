import errno
import logging
import os

from birdnet_analyzer.utils import runtime_error_handler

logger = logging.getLogger(__name__)


@runtime_error_handler
def main():
    from birdnet_analyzer import cli, segments

    parser = cli.segments_parser()
    args = parser.parse_args()

    for path in (args.audio_input, args.results):
        if path and not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    if not segments(**vars(args)):
        raise ValueError(
            "No segments were extracted: no recording in the audio folder has a "
            "result file with detections between --min_conf and --max_conf. "
            "Segments are read from the Raven selection tables, csv and Kaleidoscope "
            "results of an analysis, one combined file or one file per recording, "
            "and from Audacity labels with one file per recording; parquet results "
            "cannot be read."
        )

    logger.info("The segments are saved in %s", args.output or args.audio_input)
