.. _installation:

Download & Setup
================

GUI installer
-------------

You can download the latest BirdNET-Analyzer installer for Windows and MacOS from our `Releases page <https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest>`_. This installer provides an easy setup process for running BirdNET-Analyzer on your system. Make sure to check to select the correct installer for your system.

.. note::
    The macOS installer needs a Mac with Apple silicon (M1 or newer) and macOS 15 or later.

Windows Package Manager
-----------------------

On Windows, once the package is listed in the Windows Package Manager, you can also install and upgrade BirdNET-Analyzer with:

.. code-block:: powershell

   winget install BirdNETTeam.BirdNETAnalyzer
   winget upgrade BirdNETTeam.BirdNETAnalyzer

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

Models and the first start
--------------------------

The installers and the Linux packages contain the default BirdNET 3.0 model, including its species range model,
so the first analysis works without an internet connection. Everything else is downloaded the first time it is used and kept for later
runs, with the progress shown in the GUI: the other models (BirdNET 2.4, about 80 MB, its species range model, about
46 MB, which a location filter with BirdNET 2.4 uses, and Perch, about 380 MB), and
the model that training needs to
export a custom classifier as a TFLite or Raven model (about 320 MB for BirdNET 3.0, about 125 MB for 2.4). The first
training with one of these exports downloads it as soon as the training data are loaded. An analysis never does, and
neither does a training that saves only the detached classifier. A pip installation contains no model and downloads
the default one the same way, about 380 MB, on the first analysis or training.

The settings tab of the GUI lists every model, whether it is on the computer and how large its download is. Where a
model is chosen, a line below the choice says so when the first use is going to download it, and the location
settings of an analysis say so for the species range model.

The macOS app contains the BirdNET 2.4 model as well, so analyses with BirdNET 2.4 need no download there. Its species
range model and the model that exports a custom classifier are downloaded as described above.

The Windows installer keeps the BirdNET 2.4 model of the installation it replaces, which every release up to 2.4.0
contained: analyses and training with BirdNET 2.4 go on without a download. The species range model of 2.4 is not
among the files it can keep, so the first analysis with BirdNET 2.4 and a location filter downloads it. From an
installation of 1.x, which kept the GUI language and the last used folders inside the installation folder, it also
keeps those, for every user of the computer who has none of their own yet.

Do not install 1.5.0 or 1.5.1 over this version: their installer deletes ``%LOCALAPPDATA%\BirdNET-Analyzer-GUI``,
the folder that holds the models of this version.

The installed application keeps its models in

- ``%LOCALAPPDATA%\BirdNET-Analyzer-GUI\models`` on Windows,
- ``~/Library/Application Support/BirdNET-Analyzer-GUI/models`` on macOS,
- ``~/.local/share/BirdNET-Analyzer-GUI/models`` on Linux.

The settings tab of the GUI shows this directory and lets you choose another one, for example on a drive with more
space; models that are already downloaded stay where they are and have to be moved by hand.
Setting the environment variable ``BIRDNET_APP_DATA`` to a directory overrides both. A pip installation without
either uses the directory of the ``birdnet`` library (``%APPDATA%\birdnet``,
``~/Library/Application Support/birdnet`` or ``~/.local/share/birdnet``).

**Computers without internet access.** Models the installer does not contain have to come from a computer that is
online and runs the same BirdNET-Analyzer version. There, run one analysis with each model you need. For training,
also run one training with a TFLite or Raven export, on the BirdNET version (2.4 or 3.0) you will train on. Then copy
the whole model directory, hidden files included, into the same place on the offline computer, or onto a drive and
select that directory in the settings tab.

A pip installation can fetch the export model without a training run, in its own environment (``'2.4'`` for BirdNET
2.4):

.. code-block:: bash

   python -c "import birdnet_analyzer, birdnet; birdnet.load('acoustic', '3.0', 'pb')"

With ``BIRDNET_OFFLINE=1`` set, nothing is downloaded and a missing model is reported at once. Training reports a
missing export model once the training data are loaded, before the tuning and the training itself.

**Restricted networks.** The models come from ``zenodo.org``, ``github.com`` (which redirects to
``githubusercontent.com``) and ``tuc.cloud``; a firewall has to allow these. The GUI and the command line tools
verify downloads with the certificates of the operating system, so a company or campus proxy that inspects HTTPS
works as long as its certificate is installed there, as it is for the browser. A proxy entered in the Windows
settings is used, as are the environment variables ``HTTP_PROXY`` and ``HTTPS_PROXY``; an automatic proxy
configuration script is not.

**Uninstalling** BirdNET-Analyzer keeps the models and the settings. To remove them as well, delete the model
directory and, on Windows, ``%APPDATA%\BirdNET-Analyzer-GUI``.

Raven Pro
---------

If you want to analyze audio files without any additional coding or package install, you can now use `Raven Pro software <https://ravensoundsoftware.com/software/raven-pro/>`_ to run BirdNET models.
After download, BirdNET is available through the new "Learning detector" feature in Raven Pro.

For more information on how to use this feature, please visit the `Raven Pro Knowledge Base <https://ravensoundsoftware.com/article-categories/learning-detector/>`_.

`Download the BirdNET 2.4 model for Raven Pro here <https://tuc.cloud/index.php/s/2TX59Qda2X92Ppr/download/BirdNET_GLOBAL_6K_V2.4_Model_Raven.zip>`_, extract the zip-file and move the extracted folder to the Raven models folder. 
On Windows, the models folder is ``C:\Users\<Your user name>\Raven Pro 1.6\Models``. Start Raven Pro and select *BirdNET_GLOBAL_6K_V2.4_Model_Raven* as learning detector.

Python Package
--------------

BirdNET-Analyzer is on `PyPI <https://pypi.org/project/birdnet-analyzer/>`_ and needs Python 3.11 to 3.13:

.. code-block:: bash

   pip install birdnet_analyzer

This provides the command line tools ``birdnet-analyze``, ``birdnet-segments``, ``birdnet-species``, ``birdnet-train``
and ``birdnet-evaluate``. Additional feature sets are available as extras:

.. list-table::
   :header-rows: 1

   * - Extra
     - Adds
   * - ``embeddings``
     - embedding extraction and similarity search (``birdnet-embeddings``, ``birdnet-search``)
   * - ``train``
     - hyperparameter autotuning when training custom classifiers
   * - ``gui``
     - the graphical user interface (``birdnet-gui``), including ``embeddings`` and ``train``
   * - ``all``
     - everything above (currently the same set as ``gui``)

.. code-block:: bash

   pip install "birdnet_analyzer[all]"

On Linux, the ``gui`` extra needs the system libraries listed under `Command line installation`_.

Using BirdNET in your own Python code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run BirdNET models from your own code, install `birdnetlib <https://joeweiss.github.io/birdnetlib/>`_ or the `birdnet <https://pypi.org/project/birdnet/>`_ library, which BirdNET-Analyzer runs its analyses with:

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