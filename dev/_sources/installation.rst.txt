.. _installation:

Download & Setup
================

GUI installer
-------------

You can download the latest BirdNET-Analyzer installer for Windows and MacOS from our `Releases page <https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest>`_. This installer provides an easy setup process for running BirdNET-Analyzer on your system. Make sure to check to select the correct installer for your system.

.. note::
    | Installation was only tested on M1 and M2 chips.
    | Feedback on older Intel CPUs or newer M3 chips is welcome!

Windows Package Manager
-----------------------

On Windows, once the package is listed in the Windows Package Manager, you can also install and upgrade BirdNET-Analyzer with:

.. code-block:: powershell

   winget install BirdNET-Team.BirdNET-Analyzer
   winget upgrade BirdNET-Team.BirdNET-Analyzer

Linux
-----

Two packages are attached to each `release <https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest>`_.
Both need an x86_64 distribution with glibc 2.35 or newer, such as Ubuntu 22.04, Debian 12 or Fedora 36 and later.

- ``BirdNET-Analyzer-<version>-linux_x86_64.AppImage`` runs without installation: make the file executable and start it.
  The command line tool is included, ``./BirdNET-Analyzer-<version>-linux_x86_64.AppImage analyze --help`` shows its options.
- ``birdnet-analyzer_<version>_amd64.deb`` is for Debian, Ubuntu and their derivatives. It adds a menu entry and the
  ``birdnet-analyzer-gui`` and ``birdnet-analyze`` commands:

  .. code-block:: bash

     sudo apt install ./birdnet-analyzer_<version>_amd64.deb

Settings and the downloaded models are kept in ``~/.local/share/BirdNET-Analyzer-GUI``.

Raven Pro
---------

If you want to analyze audio files without any additional coding or package install, you can now use `Raven Pro software <https://ravensoundsoftware.com/software/raven-pro/>`_ to run BirdNET models.
After download, BirdNET is available through the new "Learning detector" feature in Raven Pro.

For more information on how to use this feature, please visit the `Raven Pro Knowledge Base <https://ravensoundsoftware.com/article-categories/learning-detector/>`_.

`Download the newest model version here <https://tuc.cloud/index.php/s/2TX59Qda2X92Ppr/download/BirdNET_GLOBAL_6K_V2.4_Model_Raven.zip>`_, extract the zip-file and move the extracted folder to the Raven models folder. 
On Windows, the models folder is ``C:\Users\<Your user name>\Raven Pro 1.6\Models``. Start Raven Pro and select *BirdNET_GLOBAL_6K_V2.4_Model_Raven* as learning detector.

Python Package
--------------

The easiest way to setup BirdNET on your machine is to install `birdnetlib <https://joeweiss.github.io/birdnetlib/>`_ or `birdnet <https://pypi.org/project/birdnet/>`_ through pip with:

.. code-block:: bash

   pip install birdnetlib

or

.. code-block:: bash

   pip install birdnet

Please take a look at the `birdnetlib user guide <https://joeweiss.github.io/birdnetlib/#using-birdnet-analyzer>`_ on how to analyze audio with `birdnetlib`. 

When using the `birdnet`-package, you can run BirdNET with:

.. code-block:: python

   import birdnet

   if __name__ == "__main__":
       model = birdnet.load("acoustic", "3.0", "onnx")
       predictions = model.predict("soundscape.wav")

For more examples and documentation, make sure to visit `pypi.org/project/birdnet/ <https://pypi.org/project/birdnet/>`_.

For any feature request or questions regarding `birdnet`, please add an issue or PR at `github.com/birdnet-team/birdnet <https://github.com/birdnet-team/birdnet>`_.

Command line installation
-------------------------

Requires Python 3.11 to 3.13.

Clone the repository

.. code-block:: bash

   git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
   cd BirdNET-Analyzer

Install the packages

.. code-block:: bash

   pip install .

.. note::

   If you also want to use the GUI, you need to install the additional packages with: ``pip install .[gui]``.
   Same goes for the training tools and the embedding/search tools: ``pip install .[train]`` and ``pip install .[embeddings]``.

   Use ``pip install .[all]`` to install all packages.

   On Linux, the ``gui`` extra installs PySide6 for the application window. Qt needs the system's xcb
   libraries; on Debian and Ubuntu, ``sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1
   libxcb-shape0 libxkbcommon-x11-0`` covers the ones a desktop installation may lack.

Verify the installation

.. code-block:: bash

   python -m birdnet_analyzer.analyze <some-audio-file.wav>