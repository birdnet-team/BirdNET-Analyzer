.. _installation:

Download & Setup
================

GUI installer
-------------

You can download the latest BirdNET-Analyzer installer for Windows and MacOS from our `Releases page <https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest>`_. This installer provides an easy setup process for running BirdNET-Analyzer on your system. Make sure to check to select the correct installer for your system.

.. note::
    | Installation was only tested on M1 and M2 chips.
    | Feedback on older Intel CPUs or newer M3 chips is welcome!

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

   from pathlib import Path
   from birdnet.models import ModelV2M4

   # create model instance for v2.4
   model = ModelV2M4()

   # predict species within the whole audio file
   species_in_area = model.predict_species_at_location_and_time(42.5, -76.45, week=4)
   predictions = model.predict_species_within_audio_file(
      Path("soundscape.wav"),
      filter_species=set(species_in_area.keys())
   )

   # get most probable prediction at time interval 0s-3s
   prediction, confidence = list(predictions[(0.0, 3.0)].items())[0]
   print(f"predicted '{prediction}' with a confidence of {confidence:.6f}")
   # predicted 'Poecile atricapillus_Black-capped Chickadee' with a confidence of 0.814056

For more examples and documentation, make sure to visit `pypi.org/project/birdnet/ <https://pypi.org/project/birdnet/>`_.

For any feature request or questions regarding `birdnet`, please add an issue or PR at `github.com/birdnet-team/birdnet <https://github.com/birdnet-team/birdnet>`_.

Command line installation
-------------------------

Requires Python 3.10. or 3.11.

Clone the repository

.. code-block:: bash

   git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
   cd BirdNET-Analyzer

Install the packages

.. code-block:: bash

   pip install .

.. note::

   If you also want to use the GUI, you need to install the additional packages with: ``pip install .[gui]``.
   Same goes for server and training tools: ``pip install .[server]`` and ``pip install .[train]``.

   Use ``pip install .[all]`` to install all packages.

   When building a GUI for systems using GTK with pywebview, you may need to install additional packages: qtpy and PyGObject. 
   Use the following command: ``pip install qtpy PyGObject``.

Verify the installation

.. code-block:: bash

   python -m birdnet_analyzer.analyze <some-audio-file.wav>