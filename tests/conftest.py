import os

# Importing the package applies the GUI's persisted model directory to the birdnet
# library; a test run must not depend on this machine's GUI settings.
if "BIRDNET_APP_DATA" not in os.environ:
    from birdnet.utils.local_data import get_app_data_path

    os.environ["BIRDNET_APP_DATA"] = str(get_app_data_path() / "birdnet")
