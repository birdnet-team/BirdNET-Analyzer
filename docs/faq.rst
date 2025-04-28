FAQ
===

We will answer frequently asked questions here. If you have a question that is not answered here, please let us know at `ccb-birdnet@cornell.edu <mailto:ccb-birdnet@cornell.edu>`_.

What is BirdNET-Analyzer?
-------------------------

BirdNET-Analyzer is a tool for analyzing bird sounds using machine learning models. It can identify bird species from audio recordings and provides various functionalities for training custom classifiers, extracting segments, and reviewing results.

How do I install BirdNET-Analyzer?
----------------------------------

BirdNET-Analyzer can be installed using different methods, including:

- | **Raven Pro**: Follow the instructions provided in the Raven Pro documentation.
- | **Python Package**: Install via pip using `pip install birdnet`.
- | **Command Line**: Download the repository and run the scripts from the command line.
- | **GUI**: Download the GUI version from the `releases page <https://github.com/birdnet-team/BirdNET-Analyzer/releases/latest>`_ and follow the installation instructions.

What licenses are used in BirdNET-Analyzer?
-------------------------------------------

BirdNET-Analyzer source code is released under the **MIT License**. The models used in the project are licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**. Please review and adhere to the specific license terms provided with each model.
Custom models trained with BirdNET-Analyzer are also subject to the same licensing terms.

.. note:: Please note that all educational and research purposes are considered non-commercial use and it is therefore freely permitted to use BirdNET models in any way.

Please get in touch if you have any questions or need further assistance. 

How can I contribute training data to BirdNET?
----------------------------------------------

The prefered way to contribute labeled audio recordings is through the `Xeno-Canto <https://www.xeno-canto.org/>`_ platform. We regularly download new recordings from Xeno-Canto to improve the models.

Fully annotated soundscape recordings should be shared on Zenodo or other data repositories - this way, they can be used for training and validation by the BirdNET team and other researchers.

If you have large amounts of validated detections, please get in touch with us at `ccb-birdnet@cornell.edu <mailto:ccb-birdnet@cornell.edu>`_.

What are the non-event classes in BirdNET?
------------------------------------------

There are currently 11 non-event classes in BirdNET:

* Human non-vocal_Human non-vocal
* Human vocal_Human vocal
* Human whistle_Human whistle
* Noise_Noise
* Dog_Dog
* Engine_Engine
* Environmental_Environmental
* Fireworks_Fireworks
* Gun_Gun
* Power tools_Power tools
* Siren_Siren

`Noise_Noise` and `Environmental_Environmental` are auxiliary classes used for training and will never be predicted by the model.