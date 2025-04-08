import os
from multiprocessing import freeze_support
import shutil
import tempfile

import birdnet_analyzer.config as cfg
import birdnet_analyzer.cli as cli
import birdnet_analyzer.utils as utils


def start_server(host="0.0.0.0", port=8080, spath="uploads/", threads=1, locale="en"):
    """
    Starts a web server for the BirdNET Analyzer.
    Args:
        host (str): The hostname or IP address to bind the server to. Defaults to "0.0.0.0".
        port (int): The port number to listen on. Defaults to 8080.
        spath (str): The file storage path for uploads. Defaults to "uploads/".
        threads (int): The number of threads to use for TensorFlow Lite inference. Defaults to 1.
        locale (str): The locale for translated labels. Defaults to "en".
    Behavior:
        - Ensures the required model files exist.
        - Loads eBird codes and labels, including translated labels if available for the specified locale.
        - Configures various settings such as file storage path, minimum confidence, result types, and temporary output path.
        - Starts a Bottle web server to handle requests.
        - Cleans up temporary files upon server shutdown.
    Note:
        This function blocks execution while the server is running.
    """
    import bottle

    import birdnet_analyzer.analyze.utils as analyze

    utils.ensure_model_exists()

    # Load eBird codes, labels
    cfg.CODES = analyze.load_codes()
    cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)

    # Load translated labels
    lfile = os.path.join(
        cfg.TRANSLATED_LABELS_PATH, os.path.basename(cfg.LABELS_FILE).replace(".txt", "_{}.txt".format(locale))
    )

    if locale not in ["en"] and os.path.isfile(lfile):
        cfg.TRANSLATED_LABELS = utils.read_lines(lfile)
    else:
        cfg.TRANSLATED_LABELS = cfg.LABELS

    # Set storage file path
    cfg.FILE_STORAGE_PATH = spath

    # Set min_conf to 0.0, because we want all results
    cfg.MIN_CONFIDENCE = 0.0

    # Set path for temporary result file
    cfg.OUTPUT_PATH = tempfile.mkdtemp()

    # Set result types
    cfg.RESULT_TYPES = ["audacity"]

    # Set number of TFLite threads
    cfg.TFLITE_THREADS = threads

    # Run server
    print(f"UP AND RUNNING! LISTENING ON {host}:{port}", flush=True)

    try:
        bottle.run(host=host, port=port, quiet=True)
    finally:
        shutil.rmtree(cfg.OUTPUT_PATH)


if __name__ == "__main__":
    # Freeze support for executable
    freeze_support()

    # Parse arguments
    parser = cli.server_parser()

    args = parser.parse_args()

    start_server(**vars(args))
