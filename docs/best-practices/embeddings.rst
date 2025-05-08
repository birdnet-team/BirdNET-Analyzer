Embedding Extraction and Search
===============================

1. Introduction 
----------------

The embeddings extraction and search feature allows you to quickly search for similar audio files in large datasets.
This can be used to explore your data and help you to collect training data for building a custom classifier.


2. Extracting Embeddings and creating a Database
-------------------------------------------------

The first step is to create a database of embeddings from your audio files.
In the GUI go to the Embeddings-Tab and then to the Extract-Section. There you can select the directory containing your audio files.
As with the analysis feature this will go to your directory recursively and include all audio files from the selected folder and any subdirectories.
After that choose the directory where your embeddings database should be created and specify a name for your database.

You can further specify the following parameters for the extraction:

- | **Overlap**: Audio is still processed in 3-second snippets. This parameter specifies the overlap between these snippets.
- | **Batch size**: This can be adjusted to increase the performance of the extraction process depending on your hardware.
- | **Audio speed modifier**:  This can be used to speed up or slow down the audio during the extraction process to enable working with ultra- and infrasonic recordings.
- | **Bandpass filter frequencies**: This sets the bandpass filter which is applied after the speed modifier, to further filter out unwanted frequencies.

.. note::
    The audio speed and bandpass parameters will be stored in the database and will also applied during the search process.

.. note::
    Due to limitations of the underlying hoplite database, multithreading is not supported for the extraction process.

The database will be created as a folder with the specified name containing two files:
    - 'hoplite.sqlite'
    - 'usearch.index'

2.1. File output
^^^^^^^^^^^^^^^^^^^

If you want to process the embeddings as files, you can also specify a folder for the file output. If no folder is specified the file output will be omitted.
The file output will create an individual file for each 3 second audio snippet that is processed, containing the extracted embedding.


3. Searching your database
-------------------------------------------------

- 