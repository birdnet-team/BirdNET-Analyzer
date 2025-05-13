Crop Modes
===============================

This page describes the different crop modes available for the training and embeddings-search feature the BirdNET-Analyzer.
In general a crop mode selection will be available in cases where audio files longer than 3 seconds are processed.
With the the crop mode you can specify how the audio files should be cropped into 3 second snippets.

1. Center
----------------

This crop mode will take the center 3 seconds of the audio file.

2. First
----------------

This crop mode will take the first 3 seconds of the audio file.

3. Segments
----------------

With this crop mode you can also specify an overlap. The crop mode will then split the audio file into 3 second segments with the specified overlap.
In the training feature this will result in multiple training examples that are generated from the same audio file.
In the search feature the similarity measure will be averaged over all segments of the query example. 


4. Smart
----------------

# TODO