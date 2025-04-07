Segment Review
=================================

Get started by listening to this AI-generated summary of segments review:

.. raw:: html

    <audio controls>
      <source src="../_static/BirdNET_Guide-Segment_review-NotebookLM.mp3" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>

| 
| `Source: Google NotebookLM`

1. Prepare Audio and Result Files
---------------------------------

- | **Collect Audio Recordings and Corresponding BirdNET Result Files**: Organize them into separate folders.
- | **Result File Formats**: BirdNET-Analyzer typically produces result files with extensions ".BirdNET.txt" or ".BirdNET.csv". It can process various result file formats, including "table", "kaleidoscope", "csv", and "audacity".
- | **Understanding Confidence Values**: Note that BirdNET confidence values are not probabilities and are not directly transferable between different species or recording conditions.

2. Using the "Segments" Function in the GUI or Command Line
-----------------------------------------------------------

- | **Segments Function**: BirdNET provides the "segments" function to create a collection of species-specific predictions that exceed a user-defined confidence value. This function is available in the graphical user interface (GUI) under the "segments" tab or via the "segments.py" script in the command line.
- | **GUI Usage**: In the GUI, you can select audio, result, and output directories. You can also set additional parameters such as the minimum confidence value, the maximum number of segments per species, the audio speed, and the segment length.

3. Setting Parameters
---------------------

- | **Minimum Confidence (min_conf)**: Set a minimum confidence value for predictions to be considered. Note that this value may vary by species. It is recommended to determine the threshold by reviewing precision and recall.
- | **Maximum Number of Segments (num_seq)**: Specify how many segments per species should be extracted.
- | **Audio Speed (audio_speed)**: Adjust the playback speed. Extracted segments will be saved with the adjusted speed (e.g., to listen to ultrsonic calls).
- | **Segment Length (seq_length)**: Define how long the extracted audio segments should be. If you set to more than 3 seconds, each segment will be padded with audio from the source recording. For example, for 5-second segment length, 1 second of audio before and after each extracted segment will be included. For 7 seconds, 2 seconds will be included, and so on. The first and last segment of each audio file might be shorter than the specified length.

4. Extracting Segments
----------------------

- | **Start the Extraction Process**: After setting all parameters, start the extraction process. BirdNET will create subfolders for each identified species and save audio clips of the corresponding recordings.
- | **Progress Display**: The progress of the process will be displayed.

5. Reviewing Results
--------------------

- | **Manual Review of Audio Segments**: The resulting audio segments can be manually reviewed to assess the accuracy of the predictions. It is important to note that BirdNET confidence values are not probabilities but a measure of the algorithm's prediction reliability.
- | **Systematic Review**: It is recommended to start with the highest confidence scores and work down to the lower scores.
- | **File Naming**: Files are named with confidence values, allowing for sorting by values.

6. Using the Review Tab in the GUI
----------------------------------

- | **Review Tab Overview**: The review tab in the GUI allows you to systematically review and label the extracted segments. It provides tools for visualizing spectrograms, listening to audio segments, and categorizing them as positive or negative detections.
- | **Collect Segments**: Use the review tab to collect segments from the specified directory. You can shuffle the segments for a randomized review process.
- | **Create Log Plot**: The review tab can generate a logistic regression plot to visualize the relationship between confidence values and the likelihood of correct detections.
- **Review Process**:

  - | **Select Directory**: Choose the directory containing the segments to be reviewed.
  - | **Species Dropdown**: Select the species to review from the dropdown menu.
  - | **File Count Matrix**: View the count of files to be reviewed, positive detections, and negative detections.
  - | **Spectrogram and Audio**: Visualize the spectrogram and listen to the audio segment.
  - | **Label Segments**: Use the buttons to label segments as positive or negative detections. You can also use the left and right arrow keys to assign labels.
  - | **Undo**: Undo the last action if needed.
  - | **Download Plots**: Download the spectrogram and regression plots for further analysis.

7. Alternative Approaches
-------------------------

- | **Raven Pro**: BirdNET result tables can be imported into Raven Pro and reviewed using the selection review function.
- | **Converting Confidence Values to Probabilities**: Another approach is converting confidence values to probabilities using logistic regression in R. However, this still requires manual evaluation of predictions.

8. Important Notes
------------------

- | **Non-Transferability of Confidence Values**: BirdNET confidence values are not easily transferable between species.
- | **Audio Quality**: The accuracy of results heavily depends on the quality of audio recordings, such as sample rate and microphone quality.
- | **Environmental Factors**: Results can be influenced by the recording environment, such as wind or rain.
- | **Standardized Test Data**: Using standardized test data for evaluation is important to make results comparable.

This guide summarizes the best practices for using the "segments" function of BirdNET-Analyzer and emphasizes the need for careful interpretation of the results.