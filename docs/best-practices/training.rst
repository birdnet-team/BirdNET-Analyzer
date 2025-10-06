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

3. Training Process
----------------------

After preparing your data you can start the training process using the BirdNET-Analyzer's training feature.
The feature can be used via the GUI or the command line interface.

In the GUI go the Train-Tab and select the directory containing your training data. The detected class names will be displayed in a table.
Further select the output directory and specify the name for your custom classifier. After that you can already start training your classifier with the default settings by clicking the "Start training" button.

3.1 Hyperparameters and Autotune
#################################

There are several hyperparameters that can be adjusted to optimize the classifier training.
If you don't have experience with training machine learning models, we recommend using the autotune feature.
This will run multiple training runs (aka trials) with different hyperparameter settings and select the best performing settings based on the validation data.
The parameters used for training the final classifier will be saved alongside the resulting classifier.
When using autotune you can specify the number of trials as well as the number of executions per trial.

If you want to adjust the hyperparameters manually, we have a more detailed documentation available :doc:`here <../implementation-details/training-hyperparameters>`.

3.2 Audio Settings
###################

When training a custom classifier you can apply a bandpass filter and also modify the speed of your audio to shift the frequency of your audio to the range of the BirdNET model.
This also enables you to train classifiers for ultra- or infrasonic signals, i.e. bats or whales.

.. caution::
   These settings also need to be applied when using the trained classifier for inference.

3.3 Caching Training Data
##########################

A majority of the training time is spent on loading the audio data and extracting the embeddings which are used for training the classifier.
To speed up the iteration of multiple training runs with same data we recommend using the caching feature. This will store the extracted embeddings in a cache file which can be loaded in later training runs. 

To create a cache file choose "save" as the "training data cache mode" in the settings and specify the location and the name for the cache file.
In later training runs you can then choose "load" as the "training data cache mode" and select the cache file you created before.


.. note::
  As the cache file contains the embeddings extracted from the audio files, all parameters that refer to the audio processing (e.g. speed modifier, bandpass filter frequencies, crop mode) can't be changed when loading the cache file.

3.4 Using test data
#####################

You can provide a separate dataset for testing your custom classifier after training is finished.
The test data should be structured in the same way as the training data, with each class having its own folder.

Precision, Recall, F1-Score, AUPRC and AUROC will be calculated for the test data.
The metrics will be calculated for each class as well as a macro-average across all classes.
Threshold based metrics will be calculated with the default threshold of 0.5 as well as an optimal threshold.

Results for the default threshold will be shown in the GUI.
The complete results, including results for the optimal threshold, will be saved to a CSV file in the output directory.

.. hint::
  The optimal threshhold is selected based on the F1-Score. This might cause precision or recall to be lower than on the default threshhold.

3.5 Model Save Mode
##########################

Custom classifiers can be saved with 2 different modes:

- **Append**: The trained classifier will extend the existing set of classes that BirdNET can detect. 
- **Replace**: The trained classifier will replace the BirdNET classifier and will only be able to detect classes provided during the training.

Choose the mode that fits your use case best, depending on whether you need to detect classes originally included in BirdNET or not.

.. caution::
   When using the "Append" mode, make sure that the class names of the new classes do not conflict with existing classes in BirdNET.

4. Using the Custom Classifier
--------------------------------

After the training process is finished your output folder should like this:

.. code-block:: text

  classifier-output/
  ├── CustomClassifier.tflite
  ├── CustomClassifier_Labels.txt
  ├── CustomClassifier_Params.csv
  └── ...

To use this classifier select the "Custom classifier" option in the species selection section of the BirdNET-Analyzer GUI and select the .tflite file.

When using the CLI you can specify the path to the .tflite file using the `-\-classifier` or `-c` argument.