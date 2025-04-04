Training Custom Classifiers
==============================================

Get started by listening to this AI-generated summary of training custom classifiers with BirdNET embeddings:

.. raw:: html

    <audio controls>
      <source src="../_static/BirdNET_Guide-Training-NotebookLM.mp3" type="audio/mpeg">
      Your browser does not support the audio element.
    </audio>

| 
| `Source: Google NotebookLM`

1. Data Collection and Preparation
----------------------------------

- | **High-Quality Audio Data**: Use recordings with minimal background noise, wind, or overlapping sounds from other species. Prefer lossless formats like WAV or FLAC over MP3 to retain important frequency details.

- | **Balanced Signal-to-Noise Ratio (SNR)**: Ensure a good balance between the target signal and background noise. A balanced SNR helps the model perform well in real-world situations.

- **Diverse and Representative Samples**:

  - Include recordings from various locations to ensure the model performs well across regions.
  - Consider seasonal and temporal variations, as bird calls can change with seasons and times of day.
  - Use data from different microphones and recording devices to make the model robust against different hardware.

- **Balanced Species Distribution**:

  - Avoid dataset biases by using a relatively balanced number of samples per species.
  - For rare species, use as many high-quality examples as possible without overfitting the model.

- | **Noise or Background Class**: Include a "noise" or "background" class. The model needs to learn what is not the target signal. This class helps the model recognize sounds that may resemble target sounds or occur in the background. Use random segments from your recordings without the target vocalizations.

- | **Organize Training Data**: Organize your training data into folders, with each folder representing a class. Folder names are used as labels.

- | **3-Second Audio Snippets**: BirdNET accepts 3-second audio snippets. If your files are shorter, they will be padded with zeros; if longer, multiple 3-second segments will be used. It may be useful to split longer recordings into shorter segments to remove non-target signal.

2. Using the BirdNET-Analyzer GUI
---------------------------------

- | **Download the GUI**: Download the BirdNET-Analyzer GUI from the website. The GUI provides an easy-to-use interface for training and analyzing audio data without needing to write code.
- | **Start the GUI**: Unzip the file and start the executable `birdnet_analyzer_gui`. This will launch the graphical interface where you can configure your training and analysis settings.
- | **Select Training Data**: In the "Training" tab, select your training data by navigating to the folder containing your class subfolders. Each subfolder should contain audio files for a specific class.
- | **Specify Output Location**: Provide a location to save the trained classifier. This is where the model will be saved after training is complete.
- | **Adjust Hyperparameters**: You can adjust hyperparameters, but default values are generally sufficient. Hyperparameters include settings like learning rate, batch size, and number of epochs.
- | **Start Training**: Start the training process. This may take some time depending on your hardware. The GUI will display progress and provide updates on the training status.

.. note::

    When adjusting low- and high-pass frequencies or modifiying the audio speed, make sure to match these setting during the analysis process.
    Custom models might underperform in Raven when changing these settings, since Raven uses different bandpass filter settings.

3. Analyzing the Data
---------------------

- | **Select Test Data**: In the "Multiple Files Processing" tab, select the folder containing your test data. This folder should contain audio files that you want to analyze using the trained classifier.
- | **Specify Output Location**: Choose a location for the output files. If not specified, output files will be saved in the same folder as the input files.
- | **Select Output Format**: Choose the output format (e.g., Raven selection tables, Audacity annotations, CSV). The output format determines how the analysis results will be saved and presented.
- | **Use Custom Classifier**: Select "Custom classifier" and navigate to the folder containing your trained classifier. This will load the custom model you trained for analyzing the test data.
- | **Start Analysis**: Begin the analysis process. The GUI will process the audio files and generate output files based on the selected format.

4. Interpreting the Results
---------------------------

- | **Review Output Files**: Check the output files (e.g., selection tables) in Raven or another bioacoustics program. These files contain the analysis results, including detected bird calls and their timestamps.
- | **Check for False Positives**: Look for false positives (detections where the model identified the target signal, but it was not present). If there are many, consider adding a noise class and retraining the model.
- | **Frequency Settings**: Ensure the frequency settings in the selection table match the frequencies of your analyzed audio data. This helps in accurately identifying bird calls within the correct frequency range.
- | **Verify Accuracy**: Listen to the audio recordings to verify the accuracy of detections. This step is crucial for validating the model's performance and ensuring reliable results.
- | **Evaluate Model Performance**: Assess the model's performance by analyzing false positives and false negatives. Identify any patterns in the errors. This evaluation helps in understanding the model's strengths and weaknesses.

5. Tips for Improving Model Performance
---------------------------------------

- | **Representative Training Data**: Ensure your training data represents the diversity of your signals. Diverse data helps the model generalize better to different environments and conditions.
- | **Use a Noise Class**: Including a noise class can significantly improve results. This class helps the model distinguish between target signals and background noise.
- | **Experiment with Settings**: Try different settings (e.g., minimum confidence threshold). Adjusting these settings can help optimize the model's performance for specific use cases.
- | **Adjust Cutoff Threshold**: If recall is low (the model misses many target vocalizations), try lowering the cutoff threshold. This can help the model detect more target signals.
- | **Add Similar Sounds to Noise Class**: If precision is low (the model produces many false positives), add sounds similar to the noise class. This helps the model better differentiate between target and non-target sounds.
- | **Use a Bandpass Filter**: Remove irrelevant frequencies with a bandpass filter. This preprocessing step can improve the model's focus on relevant frequency ranges.
- | **Use Segments**: If your training clips are longer than 3 seconds, use segments. Segmenting longer clips helps in creating consistent input data for the model.
- | **Check Diagnostic Plots**: Ensure the training process is progressing well by reviewing diagnostic plots. These plots provide insights into the model's learning curve and performance metrics.
- | **Correct File Formats and Sample Rates**: BirdNET only accepts 48 kHz inputs and rejects frequencies above 15 kHz. Ensure your audio files meet these requirements for optimal performance.

6. Additional Considerations
----------------------------

- | **Few-Shot Learning**: You can train your own model with very few examples. Few-shot learning allows the model to learn from a small number of training samples.
- | **Feature Embeddings**: BirdNET uses feature embeddings to extract relevant information for the problem. Embeddings capture important features from the audio data, which are used for classification.
- | **Quality of Embeddings**: The quality of embeddings depends on the quality of training data. High-quality training data leads to better embeddings and improved model performance.
- | **Bioacoustic Applications**: Models trained with bird sounds are often better suited for bioacoustic applications than those trained with general audio data. Specialized training data enhances the model's ability to recognize bird calls.
- | **Export to Raven**: You can export the trained classifier to Raven. This allows you to use the model within the Raven software for further analysis and visualization.
- | **Community and Support**: There is an active community and support team. Use the forum and contact the team if you have questions or feature requests. Engaging with the community can provide valuable insights and assistance.

This guide aims to help you train and improve your own models to support your research. Note that training a model is an iterative process, and you may need to try different settings and datasets to achieve the best results.