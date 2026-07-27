Docker
======

.. note::

   Image publishing was added after the ``2.4.0`` release, so **no image exists
   for 2.4.0** (and there is no ``2.4.0``/``2.4`` tag on the registry). The first
   published images will ship with the next release. Until then, or if you need
   ``2.4.0``, build the image locally (see `Building locally`_ below).

Starting with the next release, official Docker images are published to the GitHub
Container Registry on every release, for ``linux/amd64`` and ``linux/arm64``:

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

Building locally
----------------

.. code-block:: bash

   git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
   cd BirdNET-Analyzer
   docker build -t birdnet-analyzer .
