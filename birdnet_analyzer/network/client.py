"""Client to send requests to the server."""

import json
import os
import time
from multiprocessing import freeze_support

import requests


def send_request(host: str, port: int, fpath: str, mdata: str) -> dict:
    """
    Sends a classification request to the server.
    This function sends an HTTP POST request to a server for analyzing an audio file.
    It includes the audio file and additional metadata in the request payload.
    Args:
        host (str): The host address of the server.
        port (int): The port number to connect to on the server.
        fpath (str): The file path of the audio file to be analyzed.
        mdata (str): A JSON string containing additional metadata for the analysis.
        dict: The JSON-decoded response from the server.
    Returns:
        dict: The JSON-decoded response from the server.
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    url = f"http://{host}:{port}/analyze"

    print(f"Requesting analysis for {fpath}")

    # Make payload
    multipart_form_data = {"audio": (fpath.rsplit(os.sep, 1)[-1], open(fpath, "rb")), "meta": (None, mdata)}

    # Send request
    start_time = time.time()
    response = requests.post(url, files=multipart_form_data)
    end_time = time.time()

    print("Response: {}, Time: {:.4f}s".format(response.text, end_time - start_time), flush=True)

    # Convert to dict
    data = json.loads(response.text)

    return data


def _save_result(data, fpath):
    """Saves the server response.

    Args:
        data: The response data.
        fpath: The path to save the data at.
    """
    # Make directory
    dir_path = os.path.dirname(fpath)
    os.makedirs(dir_path, exist_ok=True)

    # Save result
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    import birdnet_analyzer.cli as cli

    # Freeze support for executable
    freeze_support()

    # Parse arguments
    parser = cli.client_parser()

    args = parser.parse_args()

    # TODO: If specified, read and send species list

    # Make metadata
    mdata = {
        "lat": args.lat,
        "lon": args.lon,
        "week": args.week,
        "overlap": args.overlap,
        "sensitivity": args.sensitivity,
        "sf_thresh": args.sf_thresh,
        "pmode": args.pmode,
        "num_results": args.num_results,
        "save": args.save,
    }

    # Send request
    data = send_request(args.host, args.port, args.input, json.dumps(mdata))

    # Save result
    fpath = args.output if args.output else args.i.rsplit(".", 1)[0] + ".BirdNET.results.json"

    _save_result(data, fpath)
