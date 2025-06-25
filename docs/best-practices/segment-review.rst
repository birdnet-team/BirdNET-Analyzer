Segment Review
==============

This document provides a quick overview of the segment review process in BirdNET-Analyzer, which is essential for validating species detection results.
You can also listen to an AI-generated summary of this guide in the audio player below.

.. raw:: html

    <audio controls>
      <source src="../_static/BirdNET_Guide-Segment_review-NotebookLM.mp3" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>

| 
| `Source: Google NotebookLM`

Prepare Audio and Result Files
------------------------------

The BirdNET Analyzer uses the batch analysis result tables, such as the output formats "table", "kaleidoscope" or "csv".
To collect run the batch analysis either in the graphical user interface (GUI) or :ref:`command line <cli-docs>`, which automatically creates the result files.

.. warning::

    The output format "audacity" is not supported for the segments tool since it is missing certain columns. Use "table", "kaleidoscope", or "csv" formats instead.

Using the "Segments" Tool in the GUI or Command Line
-----------------------------------------------------

The BirdNET Analyzer provides the "segments" tool to extract short audio segments from the result files and place them into separate species-specific folders.
This tool is available in the graphical user interface (GUI) under the "segments" tab or via the :ref:`birdnet_analyzer.segments <cli-segments>` script in the command line.

Setting Parameters
------------------

The GUI and command line tool allow you to set various parameters to customize the segment extraction process:

* **Minimum Confidence** (``min_conf``): Set a minimum confidence value for predictions to be considered. It is recommended to determine the threshold by reviewing precision and recall.
* **Maximum Number of Segments** (``num_seq``): Specify how many segments per species should be extracted.
* **Audio Speed** (``audio_speed``): Adjust the playback speed. Extracted segments will be saved with the adjusted speed (e.g., to listen to ultrasonic calls).
* **Segment Length** (``seq_length``): Define how long the extracted audio segments should be. If you set to more than 3 seconds, each segment will be padded with audio from the source recording. For example, for 5-second segment length, 1 second of audio before and after each extracted segment will be included. For 7 seconds, 2 seconds will be included, and so on. The first and last segment of each audio file might be shorter than the specified length.

.. note::

    The desired minimum confidence value can be different for each species.

Extracting Segments
-------------------

After setting all parameters, start the extraction process. BirdNET will create subfolders for each identified species and save audio clips of the corresponding recordings.
The progress of the process will be displayed.
The resulting audio segments will be saved in the following format:

.. code-block::

    {c}_{i}_{fname}_{start}s_{end}s.wav

where:

* ``{c}``: confidence value of the prediction (e.g., 0.835)
* ``{i}``: index of the segment inside the file
* ``{fname}``: name of the original audio file without the extension
* ``{start}``: start time of the segment inside the file in seconds
* ``{end}``: end time of the segment inside the file in seconds


Using the Review Tab in the GUI
----------------------------------

The resulting audio segments can be manually reviewed to assess the accuracy of the predictions.
It is important to note that BirdNET *confidence values are not probabilities* but a measure of the algorithm's prediction reliability.
We recommended to start with the highest confidence scores and work down to the lower scores.

The review tab in the GUI allows you to systematically review and label the extracted segments.
It provides tools for visualizing spectrograms, listening to audio segments, and categorizing them as positive or negative detections.
The review tab can generate a logistic regression plot to visualize the relationship between confidence values and the likelihood of correct detections.

In the GUI select the "Review" tab and select the segments directory you want to review.
You can now either select the parent directory containing all the different species subfolders or a specific species subfolder to review.
If you select the parent directory, the GUI will automatically select the first species subfolder, but you can switch between species via a dropdown menu.

Depending on your selection the segments will be shuffled or sorted by confidence value.
Each segment will be displayed with an audio player and its spectrogram.
After listening to a segment, you can either mark it as a positive detection (if you hear the species) or a negative detection (if you do not hear the species).
The BirdNET Analyzer will create two directories: one for positive detections and one for negative detections, and move the marked segments accordingly.
The "Undo" button allows you to revert the last action if needed.

.. note::

    You can also use the up (positive) and down (negative) arrow keys to assign labels. The left arrow key will undo the last action and the right arrow key will skip to the next segment without labeling it.

With the number of segments reviewed, the GUI will also display a logistic regression plot.
This plot shows the relationship between the confidence values and the likelihood of correct detections.
All of the plots including the spectrogram can be downloaded as PNG files for further analysis or documentation.

.. note::

    The review tab can be used on any directory containing audio files, not just those created by the segments tool. This allows you to review any set of audio files, including those from other sources.

Alternative Approaches
----------------------

- | **Raven Pro**: BirdNET result tables can be imported into Raven Pro and reviewed using the selection review function.
- | **Converting Confidence Values to Probabilities**: Another approach is converting confidence values to probabilities using logistic regression in R. However, this still requires manual evaluation of predictions.

Important Notes
---------------

- | **Non-Transferability of Confidence Values**: BirdNET confidence values are not easily transferable between species.
- | **Audio Quality**: The accuracy of results heavily depends on the quality of audio recordings, such as sample rate and microphone quality.
- | **Environmental Factors**: Results can be influenced by the recording environment, such as wind or rain.
- | **Standardized Test Data**: Using standardized test data for evaluation is important to make results comparable.

This guide summarizes the best practices for using the "segments" function of BirdNET-Analyzer and emphasizes the need for careful interpretation of the results.