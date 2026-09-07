.. _resumable-analysis:

Resumable Analysis
==================

When you analyze a directory of audio files, BirdNET-Analyzer keeps a crash-safe
journal of its progress. If the run is interrupted (by a crash, a power loss, a closed
window, or the **Pause** button in the GUI), you can continue it later instead of
starting over: files that were already analyzed are skipped, and their stored
detections are combined with the fresh results as if the run had never stopped.

How to resume a run
-------------------

Progress is stored in a hidden ``.birdnet-resume`` folder inside the output
directory (or inside the input directory, if no separate output directory was
chosen). To continue an interrupted run, simply start the same analysis again:

* **CLI**: re-run the same ``birdnet_analyzer.analyze`` command with the same
  arguments and output folder.
* **GUI**: in the batch analysis tab, select the same input and output folders. A
  status line shows how many files were already completed and the start button
  changes to **Continue analysis**.

Files that were already analyzed are skipped, the remaining files are processed, and
the output files are written for the complete set. After a successful run the
journal folder is deleted, so a finished analysis leaves nothing behind.

Which settings have to match
----------------------------

The journal records a fingerprint of every setting that affects *which detections
are produced*: the input directory, the model and its version, a custom classifier,
minimum confidence, top-N, sensitivity, overlap, the bandpass frequencies, audio
speed, latitude/longitude/week, a custom species list, the species filter threshold,
and the species-name language.

A run only continues if all of these match the interrupted run. If any of them
changed, the stored progress is discarded and the analysis starts from the beginning
— results produced with different settings are never mixed.

.. attention::

   The GUI does not restore the settings of the interrupted run when you click
   **Continue analysis** — it uses whatever is currently set in the interface. If
   you changed one of the settings listed above in the meantime, the saved progress
   is discarded and the analysis silently starts over. Loading the ``*-params.csv``
   file written next to the results restores the original settings.

Settings that only affect how results are *formatted* may change freely between the
interrupted run and its resume: the output formats, additional columns, table
splitting, merging of consecutive detections, batch size, and the number of workers
and producers. The journal stores raw detections, so the outputs are always written
with the settings of the resuming run.

Changed and added files
-----------------------

Each stored result is keyed on the input file's path *and* its size and modification
time. A file that changed on disk since it was first analyzed — edited, regenerated,
or merely touched — is re-analyzed instead of reusing the stale result. Files added
to the input directory are picked up and analyzed; stored results for files that no
longer exist are ignored.

How it works internally
-----------------------

The implementation lives in ``birdnet_analyzer/analyze/resume.py``. The journal
directory contains:

* ``manifest.json`` — the parameter fingerprint (a hash over the detection-relevant
  settings), a human-readable snapshot of those settings, the total file count, and
  the model metadata needed to write outputs when a resume finds no work left.
* ``results/<key>.parquet`` — one file per completed input file, holding that file's
  detections. ``<key>`` hashes the input path plus its size and modification time.
  Files the inference library reported as unprocessable are stored with an
  ``.invalid.parquet`` suffix so a resume does not retry them.

While an analysis runs, the ``birdnet`` library invokes an ``on_file_complete``
callback after each finished file, off the inference hot path. The callback writes
the file's detections to a temporary name and moves it into place atomically, so a
result file's presence *is* the "this file is done" marker — a crash can never leave
a half-written result that a resume would trust. The callback never raises: a
persistence problem (a full disk, say) costs the resumability of that one file, not
the running analysis.

On the next run, ``analyze()`` opens the journal before inference starts. A journal
whose fingerprint does not match the current settings is wiped and replaced. With a
matching fingerprint, the completed files are removed from the inference input, the
remaining files are analyzed (journaled the same way), and the stored detections are
merged with the fresh ones in input order — so outputs that accumulate offsets
across files, like the combined Raven table, come out exactly as they would from an
uninterrupted run. If every file was already completed, no inference happens at all
and the outputs are rebuilt from the journal alone, using the model metadata saved
in the manifest. Once the outputs are written, the journal is deleted.

The GUI builds on the same mechanism: **Pause** cancels the running inference
session while leaving the journal in place, and selecting the input/output folders
calls ``ResumeJournal.inspect()`` to display the paused progress and relabel the
start button.
