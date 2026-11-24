Training Custom Classifiers
==============================================

1. Introduction 
----------------

The training feature allows you to create custom classifiers in case BirdNET does not contain the species you are interested in.

.. note::
    Before you consider training a custom classifier you might want to check if another class can act as a proxy for detecting your species or signal of interest.
    This means if BirdNET consistently detects your target species as another class, then this class can be used in place of your target species.

2. Data Preparation
----------------------

Training data is essential for creating a custom classifier. Make sure to gather a sufficient amount of audio recordings that represent the species or signal you want to classify.
The data used for each class should be diverse and cover various conditions such as different times of day, weather conditions, and locations.

Organize your data into a directory structure where each class has its own folder containing the audio files. The directory structure should look like this:

.. code-block:: text

  dataset/
  ├── class1/
  │   ├── audio1.wav
  │   ├── audio2.wav
  │   └── ...
  ├── class2/
  │   ├── audio1.wav
  │   ├── audio2.wav
  │   └── ...
  └── ...


2.1 Non-Event Class
#####################

We recommended including a non-event class in your training data. This class should contain audio recordings that do not belong to any of the target classes and represents background noise or silence.
These classes will not be outputted when using the custom classifier, but they are essential for training the model to distinguish between target classes and non-target sounds.

The following class names can be used for the non-event samples:
  - noise
  - other
  - background
  - silence

2.2 Audio File Length
#####################

BirdNET will process 3-second audio segments from your recordings and we recommend using 3-second audio files for training.
In case your audio files are longer than 3 seconds, you can specify a crop mode to choose how these audio files are processed. See :doc:`crop modes <../implementation-details/crop-modes>` for more details. 

The training holds back a part of the files of each class to validate on, and it keeps all samples of a file together, so the validation score shows how the classifier does on recordings it has not been trained on.
A class therefore needs at least two files to be validated: a class with a single file is only trained on, and a training in which no class has a second file stops with an error.

3. Training Process
----------------------

After preparing your data you can start the training process using the BirdNET-Analyzer's training feature.
The feature can be used via the GUI or the command line interface.

In the GUI go the Train-Tab and select the directory containing your training data. The detected class names will be displayed in a table.
Further select the output directory and specify the name for your custom classifier. After that you can already start training your classifier with the default settings by clicking the "Start training" button.

3.1 Base Model
###############

A custom classifier is trained on the embeddings of a BirdNET model and only works together with the version it was trained on.
You can choose between BirdNET 3.0 (the default) and BirdNET 2.4 with the "Base model" setting in the GUI or ``--birdnet`` on the command line.

The base model also sets the audio processing: BirdNET 3.0 works on 32 kHz audio, BirdNET 2.4 on 48 kHz, both in 3-second segments.

.. note::
  A training data cache (see 3.4) holds the embeddings of one base model. Loading a cache trains on the version it was created with, whatever base model is selected.

3.2 Training Settings
######################

The training uses fixed settings that were chosen by comparing training runs on real recordings with both base models.
The learning rate depends on the base model, 0.1 for BirdNET 3.0 and 0.01 for BirdNET 2.4, and the classifier trains for up to 50 epochs.
Training stops early once the validation score no longer improves, so most trainings end well before epoch 50 with a "Stopped early" message. That is expected.

Changing the other hyperparameters rarely improved a classifier in that comparison. Autotune, which tries 50 combinations of settings, was compared on BirdNET 3.0 only and matched the fixed settings on most datasets at many times the training time.
The GUI therefore trains with the fixed settings only. The hyperparameters and autotune remain available on the command line and in the Python API; they are described :doc:`here <../implementation-details/training-hyperparameters>`.

The parameters used for training are saved alongside the resulting classifier.

3.3 Audio Settings
###################

When training a custom classifier you can apply a bandpass filter and also modify the speed of your audio to shift the frequency of your audio to the range of the BirdNET model.
This also enables you to train classifiers for ultra- or infrasonic signals, i.e. bats or whales.

.. caution::
   These settings also need to be applied when using the trained classifier for inference.

3.4 Caching Training Data
##########################

A majority of the training time is spent on loading the audio data and extracting the embeddings which are used for training the classifier.
To speed up the iteration of multiple training runs with same data we recommend using the caching feature. This will store the extracted embeddings in a cache file which can be loaded in later training runs. 

To create a cache file choose "save" as the "training data cache mode" in the settings and specify the location and the name for the cache file.
In later training runs you can then choose "load" as the "training data cache mode" and select the cache file you created before.


.. note::
  As the cache file contains the embeddings extracted from the audio files, all parameters that refer to the audio processing (e.g. speed modifier, bandpass filter frequencies, crop mode) can't be changed when loading the cache file.

3.5 Using test data
#####################

You can provide a separate dataset for testing your custom classifier after training is finished.
The test data should be structured in the same way as the training data, with each class having its own folder.
It is only used for this evaluation: autotune and early stopping always validate on the training data, so the test results stay an independent estimate.

Precision, Recall, F1-Score, AUPRC and AUROC will be calculated for the test data.
The metrics will be calculated for each class as well as a macro-average across all classes.
Threshold based metrics will be calculated with the default threshold of 0.5 as well as an optimal threshold.

Results for the default threshold will be shown in the GUI.
The complete results, including results for the optimal threshold, will be saved to a CSV file in the output directory.

.. hint::
  The optimal threshhold is selected based on the F1-Score. This might cause precision or recall to be lower than on the default threshhold.

3.6 Model Save Mode
##########################

Custom classifiers can be saved with 2 different modes:

- **Append**: The trained classifier will extend the existing set of classes that BirdNET can detect. 
- **Replace**: The trained classifier will replace the BirdNET classifier and will only be able to detect classes provided during the training.

Choose the mode that fits your use case best, depending on whether you need to detect classes originally included in BirdNET or not.

.. caution::
   When using the "Append" mode, make sure that the class names of the new classes do not conflict with existing classes in BirdNET.

3.7 Detached Classifier
##########################

You can additionally choose to save a detached version of the trained classifier.
This will save the classification head in tflite and saved model (pb) format.
Instead of audio the classification head will take the embeddings extracted from the BirdNET model as input (1024 values for BirdNET 2.4, 1280 for BirdNET 3.0).
On the 3.0 base the detached classifier outputs probabilities, on 2.4 logits, matching the respective BirdNET model.
This is meant as a way to provide a lightweight version of the classifier that can more easily be transferred to edge devices in the field that already run BirdNET. 

.. note::
  As of now the detached classifier feature is meant for custom inference pipelines and the BirdNET-Analyzer does not currently offer a way to use the detached classifier directly.

.. caution::
   The "Append" mode will not be applied to the detached classifier. The detached classifier will only contained the newly trained classes.
   If the original BirdNET classifications are needed they should be extracted from the BirdNET model alongside the embeddings that are fed into the detached classifier.

4. Using the Custom Classifier
--------------------------------

After the training process is finished your output folder should like this:

.. code-block:: text

  classifier-output/
  ├── CustomClassifier.tflite
  ├── CustomClassifier_Labels.txt
  ├── CustomClassifier.birdnet.train-params.csv
  └── ...

To use this classifier select the "Custom classifier" option in the model selection section of the BirdNET-Analyzer GUI and select the .tflite file.

When using the CLI you can specify the path to the .tflite file using the `-\-classifier` or `-c` argument.

.. note::
  The analysis reads the base model version from the classifier file, so classifiers trained on either base can be used. A classifier trained on 3.0 processes one segment at a time; the batch size setting is reduced to 1 for it.