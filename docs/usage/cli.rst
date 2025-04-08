Command line interface
======================

.. _cli-docs:

birdnet_analyzer.analyze
------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.analyzer_parser
   :prog: birdnet_analyzer.analyze

   Run ``analyzer.py`` to analyze an audio file.
   You need to set paths for the audio file and selection table output. Here is an example:

   .. code:: bash

      python -m birdnet_analyzer.analyze /path/to/audio/folder -o /path/to/output/folder

   Here are two example commands to run this BirdNET version:

   .. code:: bash

      python3 -m birdnet_analyzer.analyze example/ --slist example/ --min_conf 0.5 --threads 4

      python3 -m birdnet_analyzer.analyze example/ --lat 42.5 --lon -76.45 --week 4 --sensitivity 1.0

birdnet_analyzer.client
------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.client_parser
   :prog: birdnet_analyzer.client

   This script will read an audio file, generate metadata from command line arguments and send it to the server.
   The server will then analyze the audio file and send back the detection results which will be stored as a JSON file.

birdnet_analyzer.embeddings
---------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.embeddings_parser
   :prog: birdnet_analyzer.embeddings

   Run ``embeddings.py`` to extract feature embeddings instead of class predictions.
   Result file will contain timestamps and lists of float values representing the embedding for a particular 3-second segment.
   Embeddings can be used for clustering or similarity analysis. Here is an example:

   .. code:: bash

      python -m birdnet_analyzer.embeddings example/ --threads 4 --batch_size 16

.. _cli-segments:

birdnet_analyzer.segments
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.segments_parser
   :prog: birdnet_analyzer.segments

   After the analysis, run ``segments.py`` to extract short audio segments for species detections to verify results.
   This way, it might be easier to review results instead of loading hundreds of result files manually.

.. _cli-species:

birdnet_analyzer.species
-------------------------

The year-round list may contain some species, that are not included in any list for a specific week. See `birdnet-team#211 <https://github.com/birdnet-team/BirdNET-Analyzer/issues/211#issuecomment-1849833360>`_ for more details.

.. argparse::
   :ref: birdnet_analyzer.cli.species_parser
   :prog: birdnet_analyzer.species

birdnet_analyzer.server
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.server_parser
   :prog: birdnet_analyzer.server

   You can host your own analysis service and API by launching the ``birdnet_analyzer.server`` script.
   This will allow you to send files to this server, store submitted files, analyze them and send detection results back to a client.
   This could be a local service, running on a desktop PC, or a remote server.
   The API can be accessed locally or remotely through a browser or Python client (or any other client implementation).

   Install one additional package with ``pip install bottle``.

   Start the server with ``python -m birdnet_analyzer.server``.
   You can also specify a host name or IP and port number, e.g., ``python -m birdnet_analayzer.server --host localhost --port 8080``.

   The server is single-threaded, so you’ll need to start multiple instances for higher throughput. This service is intented for short audio files (e.g., 1-10 seconds).

   Query the API with a client.
   You can use the provided Python client or any other client implementation.
   Request payload needs to be ``multipart/form-data`` with the following fields:
   ``audio`` for raw audio data as byte code, and ``meta`` for additional information on the audio file.
   Take a look at our example client implementation in the ``client.py`` script.

   Parse results from the server. The server will send back a JSON response with the detection results. The response also contains a msg field, indicating success or error. Results consist of a sorted list of (species, score) tuples.

   This is an example response:

   .. code:: json

      {
         "msg": "success",
         "results": [
            [
                  "Poecile atricapillus_Black-capped Chickadee",
                  0.7889
            ],
            [
                  "Spinus tristis_American Goldfinch",
                  0.5028
            ],
            [
                  "Junco hyemalis_Dark-eyed Junco",
                  0.4943
            ],
            [
                  "Baeolophus bicolor_Tufted Titmouse",
                  0.4345
            ],
            [
                  "Haemorhous mexicanus_House Finch",
                  0.2301
            ]
         ]
      }
   

birdnet_analyzer.train
-------------------------

.. argparse::
   :ref: birdnet_analyzer.cli.train_parser
   :prog: birdnet_analyzer.train

   You can train your own custom classifier on top of BirdNET.
   This is useful if you want to detect species that are not included in the default species list.
   You can also use this to train a classifier for a specific location or season.
   
   All you need is a dataset of labeled audio files, organized in folders by species (we use folder names as labels).
   This also works for non-bird species, as long as you have a dataset of labeled audio files.
   
   Audio files will be resampled to 48 kHz and converted into 3-second segments (we support different crop segemnattion modes for files longer than 3 seconds; we pad with random noise if the file is shorter). We recommend using at least 100 audio files per species (although training also works with less data).
   
   You can download a sample training data set `here <https://drive.google.com/file/d/16hgka5aJ4U69ane9RQn_quVmgjVY2AY5/edit>`_.

   1. Collect training data and organize in folders based on species names.
   2. Species labels should be in the format ``<scientific name>_<species common name>`` (e.g., ``Poecile atricapillus_Black-capped Chickadee``), but other formats work as well.
   3. It can be helpful to include a non-event class. If you name a folder 'Noise', 'Background', 'Other' or 'Silence', it will be treated as a non-event class.
   4. Run the training script with ``python birdnet_analyzer.train <path to training data folder> -o <path to trained classifier model output>``.

   **The script saves the trained classifier model based on the best validation loss achieved during training. This ensures that the model saved is optimized for performance according to the chosen metric.**

   After training, you can use the custom trained classifier with the ``--classifier`` argument of the ``analyze.py`` script.
   If you want to use the custom classifier in Raven, make sure to set ``--model_format raven``.

   .. note::
      Adjusting hyperparameters (e.g., number of hidden units, learning rate, etc.) can have a big impact on the performance of the classifier.
      We recommend trying different hyperparameter settings. If you want to automate this process, you can use the ``--autotune`` argument (in that case, make sure to install ``keras_tuner`` with ``pip install keras-tuner``).

   **Example usage** (when downloading and unzipping the sample training data set):

   .. code:: bash

      python -m birdnet_analyzer.train train_data/ -o checkpoints/custom/Custom_Classifier.tflite
      python -m birdnet_analyzer.analyze example/ --classifier checkpoints/custom/Custom_Classifier.tflite

   .. note::
      Setting a custom classifier will also set the new labels file. Due to these custom labels, the location filter and locale will be disabled.
   
   **Negative samples**

   You can include negative samples for classes by prefixing the folder names with a '-' (e.g., ``-Poecile atricapillus_Black-capped Chickadee``).
   Do this with samples that definitely do not contain the species.
   Negative samples will only be used for training and not for validation.
   Also keep in mind that negative samples will only be used when a corresponding folder with positive samples exists.
   Negative samples cannot be used for binary classification, instead include these samples in the non-event folder.

   **Multi-label data**

   To train with multi-label data separate the class labels with commas in the folder names (e.g., ``Poecile atricapillus_Black-capped Chickadee,Cardinalis cardinalis_Northern Cardinal``).
   This can also be combined with negative samples as described above.
   The validation split will be performed combination of classes, so you might want to ensure sufficient data for each combination of classes.
   When using multi-label data the upsampling mode will be limited to 'repeat'.

   .. note:: Custom classifiers trained with BirdNET-Analyzer are licensed under the `Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0) <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.