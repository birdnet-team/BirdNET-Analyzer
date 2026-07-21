Docker
======

Official Docker images are published to the GitHub Container Registry with every release, for ``linux/amd64`` and ``linux/arm64``:

.. code-block:: bash

   docker pull ghcr.io/birdnet-team/birdnet-analyzer:latest

Version tags follow the GitHub releases, so ``2.4.0`` and ``2.4`` point to that release while ``latest`` always points to the most recent one.

Usage
-----

The image runs the command line interface. Mount your audio data into the container and pass the usual command line arguments:

.. code-block:: bash

   # Analyze audio files in the current directory
   docker run --rm -v "$PWD:/audio" ghcr.io/birdnet-team/birdnet-analyzer -m birdnet_analyzer.analyze /audio -o /audio/output

Any of the CLI entry points can be used, e.g. ``-m birdnet_analyzer.species`` or ``-m birdnet_analyzer.segments``.

Building locally
----------------

.. code-block:: bash

   git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
   cd BirdNET-Analyzer
   docker build -t birdnet-analyzer .
