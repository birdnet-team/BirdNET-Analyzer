Models
======


V2.4, June 2023
---------------

* more than 6,000 species worldwide
* covers frequencies from 0 Hz to 15 kHz with two-channel spectrogram (one for low and one for high frequencies)
* 0.826 GFLOPs, 50.5 MB as FP32
* enhanced and optimized metadata model
* global selection of species (birds and non-birds) with 6,522 classes (incl. 11 non-event classes)

Technical Details
^^^^^^^^^^^^^^^^^

* 48 kHz sampling rate (we up- and downsample automatically and can deal with artifacts from lower sampling rates)
* we compute 2 mel spectrograms as input for the convolutional neural network:

    * first one has fmin = 0 Hz and fmax = 3000; nfft = 2048; hop size = 278; 96 mel bins
    * second one has fmin = 500 Hz and fmax = 15 kHz; nfft = 1024; hop size = 280; 96 mel bins

* both spectrograms have a final resolution of 96x511 pixels
* raw audio will be normalized between -1 and 1 before spectrogram conversion
* we use non-linear magnitude scaling as mentioned in `Schlüter 2018 <http://ceur-ws.org/Vol-2125/paper_181.pdf>`_
* V2.4 uses an EfficienNetB0-like backbone with a final embedding size of 1024
* See `this comment <https://github.com/birdnet-team/BirdNET-Analyzer/issues/177#issuecomment-1772538736>`_ for more details

Species range model V2.4 - V2, Jan 2024
---------------------------------------

* updated species range model based on eBird data
* more accurate (spatial) species range prediction
* slightly increased long-tail distribution in the temporal resolution 
* see `this discussion post <https://github.com/birdnet-team/BirdNET-Analyzer/discussions/234>`_ for more details


Using older models
------------------

Older models can also be used as custom classifiers in the GUI or using the `--classifier` argument in the `birdnet_analyzer.analyze` command line interface.

Just download your desired model version and unzip.

* GUI: Select the \*_Model_FP32.tflite file under **Species selection / Custom classifier**
* CLI: ``python -m birdnet_analyzer ... --classifier 'path_to_Model_FP32.tflite'``

Model Version History
---------------------

.. note:: All models listed here are licensed under the `Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0) <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.

V2.4
^^^^

- more than 6,000 species worldwide
- covers frequencies from 0 Hz to 15 kHz with two-channel spectrogram (one for low and one for high frequencies)
- 0.826 GFLOPs, 50.5 MB as FP32
- enhanced and optimized metadata model
- global selection of species (birds and non-birds) with 6,522 classes (incl. 11 non-event classes)
- Download here: `BirdNET-Analyzer-V2.4.zip <https://drive.google.com/file/d/1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3>`_

V2.3
^^^^

- slightly larger (36.4 MB vs. 21.3 MB as FP32) but smaller computational footprint (0.698 vs. 1.31 GFLOPs) than V2.2
- larger embedding size (1024 vs 320) than V2.2 (hence the bigger model)
- enhanced and optimized metadata model
- global selection of species (birds and non-birds) with 3,337 classes (incl. 11 non-event classes)
- Download here: `BirdNET-Analyzer-V2.3.zip <https://drive.google.com/file/d/1hhwQBVBngGnEhmqYeDksIW8ZY1FJmwyi>`_

V2.2
^^^^

- smaller (21.3 MB vs. 29.5 MB as FP32) and faster (1.31 vs 2.03 GFLOPs) than V2.1
- maintains same accuracy as V2.1 despite more classes
- global selection of species (birds and non-birds) with 3,337 classes (incl. 11 non-event classes)
- Download here: `BirdNET-Analyzer-V2.2.zip <https://drive.google.com/file/d/166w8IAkXGKp6ClKb8vaniG1DmOr8Fwem>`_

V2.1
^^^^

- same model architecture as V2.0
- extended 2022 training data
- global selection of species (birds and non-birds) with 2,434 classes (incl. 11 non-event classes)
- Download here: `BirdNET-Analyzer-V2.1.zip <https://drive.google.com/file/d/15cvPiezn_6H2tQs1FGMVrVdqiwLjLRms>`_

V2.0
^^^^

- same model design as 1.4 but a bit wider
- extended 2022 training data
- global selection of species (birds and non-birds) with 1,328 classes (incl. 11 non-event classes)
- Download here: `BirdNET-Analyzer-V2.0.zip <https://drive.google.com/file/d/1h2Tbk_29ghNdK62ynrdRWyxT4H1fpFGs>`_

V1.4
^^^^

- smaller, deeper, faster
- only 30% of the size of V1.3
- still linear spectrogram and EfficientNet blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.4.zip <https://drive.google.com/file/d/1h14-Y8dOrPr9XCWfIoUjlWMJ9aWyNkKa>`_

V1.3
^^^^

- Model uses linear frequency scale for spectrograms
- uses V2 fusion blocks and V1 efficient blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.3.zip <https://drive.google.com/file/d/1h0nJzPjyJWbkfPyaWpS332xUwzDOygs9>`_

V1.2
^^^^

- Model based on EfficientNet V2 blocks
- uses V2 fusion blocks and V1 efficient blocks
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.2.zip <https://drive.google.com/file/d/1h-il_W6t8Tz_XHrRMO1zcp_ThYp9QPLK>`_

V1.1
^^^^

- Model based on Wide-ResNet (aka "App model")
- extended 2021 training data
- 1,133 birds and non-birds for North America and Europe
- Download here: `BirdNET-Analyzer-V1.1.zip <https://drive.google.com/file/d/1gzpwiCAf2HkfcAmlRq1K9Q0KrDsd5nGP>`_

App Model
^^^^^^^^^

- Model based on Wide-ResNet
- ~3,000 species worldwide
- currently deployed as BirdNET app model
- Download here: `BirdNET-Analyzer-App-Model.zip <https://drive.google.com/file/d/1gxkxPFlaTYxHFqAODDHYGUX8uEkZDWaL>`_
