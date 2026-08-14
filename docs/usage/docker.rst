Docker
======

.. note::

   Image publishing was added after the ``2.4.0`` release, so **no image exists for
   2.4.0** — the registry has no ``2.4.0`` or ``2.4`` tag. To run ``2.4.0`` itself,
   build the image locally (see `Building locally`_ below).

Official Docker images are published to the GitHub Container Registry for every
release after ``2.4.0``, for ``linux/amd64`` and ``linux/arm64``:

.. code-block:: bash

   docker pull ghcr.io/birdnet-team/birdnet-analyzer:latest

Each release publishes version tags following the GitHub release (``<major>.<minor>.<patch>``
and ``<major>.<minor>``, e.g. ``2.5.0`` and ``2.5``), while ``latest`` always points to
the most recent release.

Usage
-----

The image runs the command line interface. Mount your audio data into the container and pass the usual command line arguments:

.. code-block:: bash

   # Analyze audio files in the current directory
   docker run --rm -v "$PWD:/audio" ghcr.io/birdnet-team/birdnet-analyzer -m birdnet_analyzer.analyze /audio -o /audio/output

Any of the CLI entry points can be used, e.g. ``-m birdnet_analyzer.species`` or ``-m birdnet_analyzer.segments``.

The acoustic and geo models are baked into the image, so containers start analyzing
immediately and run without network access:

.. code-block:: bash

   docker run --rm --network none -v "$PWD:/audio" ghcr.io/birdnet-team/birdnet-analyzer -m birdnet_analyzer.analyze /audio -o /audio/output

Building locally
----------------

.. code-block:: bash

   git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
   cd BirdNET-Analyzer
   docker build -t birdnet-analyzer .
