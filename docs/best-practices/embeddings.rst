Embedding Extraction and Search
===============================

1. Introduction 
----------------

The embeddings extraction and search feature allows you to quickly search for similar audio files in large datasets.
This can be used to explore your data and help you to collect training data for building a custom classifier.


2. Extracting Embeddings and creating a Database
-------------------------------------------------

The first step is to create a database of embeddings from your audio files.
In the GUI go to the **Embeddings** tab. There you can select the directory containing your audio files.
As with the analysis feature this will go to your directory recursively and include all audio files from the selected folder and any subdirectories.
After that select the directory that holds the database; for a new database, choose an empty one.

You can further specify the following parameters for the extraction:

- | **Overlap**: Audio is still processed in 3-second snippets. This parameter specifies the overlap between these snippets.
- | **Batch size**: This can be adjusted to increase the performance of the extraction process depending on your hardware.
- | **Audio speed modifier**:  This can be used to speed up or slow down the audio during the extraction process to enable working with ultra- and infrasonic recordings.
- | **Bandpass filter frequencies**: This sets the bandpass filter which is applied after the speed modifier, to further filter out unwanted frequencies.

.. note::
    The audio speed and bandpass parameters will be stored in the database and will also applied during the search process.

.. note::
    A new database is filled with the embeddings of BirdNET 3.0. An existing database keeps the model it was created with, shown next to it in the GUI,
    for adding recordings and for searching; if that is BirdNET 2.4, the model is downloaded when it is first needed.
    A database written by BirdNET-Analyzer 2.x is converted the first time it is opened, and its previous ``hoplite.sqlite`` is kept as ``hoplite.sqlite.v2-backup``.

The database will be created in the selected folder, which then contains two files:
    - 'hoplite.sqlite'
    - 'usearch.index'

2.1. File output
^^^^^^^^^^^^^^^^^^^

If you want to process the embeddings as a file, check **Enable file output**. After the extraction, all embeddings of the database are then also written to a CSV file, ``embeddings.csv`` in the database folder unless you select another file.
Each row holds the path of the audio file relative to the folder that holds the input directory, the start and end of the segment in seconds and the embedding.

3. Searching your database
-------------------------------------------------

The database can be searched in the **Search** tab of the GUI.
To start the search first select the database you want to search in. The audio root, the folder that holds the audio directory the database was extracted from, is filled in from the database; select another one if the recordings have moved. As soon as the database is loaded the extraction settings and the number of embeddings in the database will be displayed.

After that select a file as a query example to find similar sounds in the database.
You can select the :doc:`crop mode <../implementation-details/crop-modes>` to specify how the example file will be cropped if it is longer than 3 seconds. For the segments crop mode the simliarity measure will be averaged over all segments of the query example.

Further specify the maximum number of results you want to retrieve and the score-function.

The following score functions are available:

- | **Cosine**: Uses the cosine of the angle between the two embedding vectors. More similar vectors will result in a higher value.
- | **Dot**: Uses the dot product of the two embedding vectors. As with the cosine measure, more similar vectors will result in a higher value, but longer vectors will also increase the score.
- | **Euclidean**: Uses the euclidean distance between the two embedding vectors. More similar vectors will result in a lower value.

Click the start search button to show the results. The results will be displayed over multiple pages.
For each result a spectrogram of the corresponding audio snippet will be shown and the audio can be played back for reference.

While inspecting the results you can mark them for export. After finishing the selection click the export button and choose a folder to save the results in.
You can now use the data for training custom classifiers or for further analysis.


3.1 About audio speed and bandpass filter in search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The audio speed and bandpass filter settings that were used for the extraction process will also be applied to the query example and the result-snippets.
This will also be reflected in the spectrograms and the audio playback.

However, the exported audio will always be in the original speed and without any bandpass filter applied.
This is to ensure that the exported audio retains its original sample rate and also to make using it in the training process more intuitive.

As the processed audio snippets are always 3 seconds long after the audio speed modifier, the exported audio files may be shorter or longer.
For example with an audio speed modifier of 2, e.g. double speed, the exported audio will only be 1.5 seconds long.

3.2. Search via CLI
^^^^^^^^^^^^^^^^^^^

Searching the database can also be done via the command line interface, although due to the missing interface a manual inspection and selection of the results is not possible.
See :ref:`birdnet_analyzer.search <cli-search>` for how to use it.

