Segment Collection Modes
===============================

This page describes the different segment collection modes available for the segment-extraction feature in the BirdNET-Analyzer.
In general the segments feature collects all detections from the provided result files according to the specified confidence range.
Then segments are selected for each species up to the specified maximum number segments.
If there are more detections than the maximum for a species, the segment collection mode gives you control over which segments are selected.
The goal of this is to get a more representative set of segments to use in the review feature or to get only high confidence segments to use as training data.
The following segment collection modes are available:

1. Random
----------------

This mode will select segments randomly from the detections that are within the specified confidence range.
Therefore it will mirror the distribution of the confidence values in the detections.

2. Confidence
----------------

This mode will select segments based on confidence values, starting with the highest confidence scores.

3. Balanced
----------------

This mode will select segments equally distributed across the specified confidence range.
In more detail, we divide the confidence range into a specified number of bins (10 by default) and sort the detections into them.
Then total number of segments is divided by the number of bins to get the maximum number of segments per bin.
Finally for each bin we randomly select the up to the maximum number of segments from each bin.
Due to rounding and the distribution of confidence values, the total number of segments might be less than the specified maximum even if there are more segments available.