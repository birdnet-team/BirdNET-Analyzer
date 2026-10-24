Breaking changes
================

This is a major release. Its central change is that analyses now use the **BirdNET
3.0** acoustic and geolocation models by default, which is not backwards compatible
with the previous release in several ways. Breaking changes are expected for a major
release; this page lists every user-visible incompatibility relative to the previous
release (**v2.4.0**) and what to do about it.

.. contents::
   :local:
   :depth: 1


Default model is now BirdNET 3.0
--------------------------------

**What changed.** Analyses use the BirdNET 3.0 acoustic model by default, and
location-based species lists always use the 3.0 geolocation model. v2.4 was the
previous default for both.

**Consequence.** Results are not directly comparable to v2.4 output: 3.0 is a different
model with different confidence scores, a larger species set (11,560 labels) and an
updated taxonomy. Do not mix or compare 3.0 result files with v2.4 ones.

**What to do.** To keep the previous acoustic model, select the 2.4 model explicitly:

.. code-block:: bash

   birdnet-analyze INPUT --birdnet 2.4

In the GUI, choose *BirdNET 2.4* under model selection. The 2.4 model remains fully
supported. Location-based species lists come from the 3.0 geolocation model either way.


Inference runs on ONNX, not TensorFlow, on the default path
-----------------------------------------------------------

**What changed.** The default 3.0 analysis and geolocation paths run on
``onnxruntime`` instead of TensorFlow, which is now a required dependency.

**Consequence.**

* Default analyses are faster on CPU and no longer load TensorFlow.
* TensorFlow is still required — and still installed — for the 2.4 model, the Perch
  model, custom classifiers, and training.
* The Docker image is substantially larger: it bakes the ~520 MB 3.0 ONNX model
  (the 2.4 TFLite model was ~120 MB).

**What to do.** Reinstall the package so ``onnxruntime`` is pulled in (your usual
``pip install`` step). Nothing changes for the 2.4, Perch, custom-classifier or
training paths.


Custom species lists and the 3.0 taxonomy
-----------------------------------------

**What changed.** The 3.0 model uses an updated taxonomy, so labels from a v2.4-era
species list may no longer match verbatim — for example ``Accipiter cooperii`` became
``Astur cooperii``, and several common names changed (``Rock Pigeon`` → ``Rock Dove``,
``Common Raven`` → ``Northern Raven``, ``Herring Gull`` → ``European Herring Gull``,
``European Starling`` → ``Common Starling``).

A species list passed with ``--slist`` is now **reconciled** to the loaded model
instead of having to match it exactly: each entry is matched by its exact label, then
its scientific name, then its common name. This carries most legacy lists over
automatically.

**Consequence.**

* Species that still cannot be matched (genuinely absent from the model, or a typo)
  are **skipped with a warning**; v2.4.0 skipped them silently.
* If **none** of the list's species can be matched, the analysis **errors**; v2.4.0
  produced no detections.
* ``--slist`` together with ``--lat`` and ``--lon`` is an error; v2.4.0 ignored the
  list.

**What to do.** Legacy lists generally work unchanged. To require that every entry
match — turning any unmatched species into a hard error — use the new ``--strict``
flag:

.. code-block:: bash

   birdnet-analyze INPUT --slist my_list.txt --strict

In the GUI, selecting a species-list file shows a warning if it contains species the
chosen model does not have (only once that model has been downloaded).


Label languages
---------------

**What changed.** The ``--locale`` codes depend on the model version. Compared with
v2.4.0:

* ``en``, ``el``, ``he``, ``in``, ``is``, ``ml`` and ``pt_BR`` are no longer accepted.
  English is ``en_us``, the new default.
* The 3.0 model has no ``af``, ``ar``, ``en_uk``, ``hu``, ``it``, ``ko``, ``ro``,
  ``sl`` or ``th``.
* ``bg``, ``ca``, ``hr``, ``lt``, ``pt_pt`` and ``sr`` are only available with the 3.0
  model, no longer with 2.4.
* New with 3.0: ``cy``, ``es_ec``, ``es_es``, ``es_mx`` and ``fa``.

**Consequence.** The command line rejects a code it does not know. A known code the
selected model version lacks falls back to ``en_us``, with a warning. For example,
``--locale it`` with the default 3.0 model produces English labels.

**What to do.** Use a locale the model version supports, or switch model versions for
one only the other has. In the GUI the locale dropdown offers only the selected model's
languages.


Training defaults to the 3.0 base
---------------------------------

**What changed.** ``birdnet_analyzer.train`` takes ``--birdnet {3.0,2.4}`` (the train
tab a "Base model" choice) and trains on BirdNET 3.0 unless told otherwise. All output
formats (tflite, Raven, detached) are available on both bases. A classifier only works
with the version it was trained on; that version is recorded in the
``*.birdnet.train-params.csv`` and in training data caches, and a cache is always
trained on the version it was created with.

**Consequence.** Classifiers trained on 3.0 process one segment at a time
(``--batch_size`` is reduced to 1). Pass ``--birdnet 2.4`` to keep training 2.4
classifiers, which behave as before. The analysis reads the version from the
classifier file, so both kinds are used the same way.

The default output location for a trained classifier is now
``checkpoints/custom/Custom_Classifier`` **relative to the current working directory**
(previously an absolute path inside the installed package). Scripts that relied on the
old install-directory location will now write to, or look in, a different place; pass an
explicit ``-o/--output`` to pin a location. The new default is writable on a
pip-installed package and survives upgrades.


Training validates on the training data
---------------------------------------

**What changed.** ``--test_data`` now only evaluates the trained classifier. A
validation split (``--val_split``, which must be above 0) is always taken from the
training data, and early stopping watches ``--autotune_metric`` (AUPRC by default)
instead of the validation loss. The AUPRC and AUROC reported during training no longer
apply the sigmoid twice, and ``--label_smoothing`` uses the standard formula.

**Consequence.** With ``--test_data``, the classifier trains on the training data minus
the validation split; v2.4.0 trained on all of it and validated on the test data. The
same command trains a different classifier than v2.4.0, and its validation scores are
not comparable.

**What to do.** Choose ``--val_split`` for the share of the training data to hold out;
``--val_split 0`` is rejected. Compare scores only between runs of this version.


Locations and species lists
---------------------------

**What changed.**

* ``-1`` no longer means "not set". ``--lat -1 --lon -1`` is a location in the Gulf of
  Guinea, and ``--week -1`` with a location is an error. Leave an option out instead.
* ``analyze`` and ``species`` reject only one of ``--lat`` and ``--lon``; v2.4.0 used
  -1 for the missing one.
* ``birdnet_analyzer.species`` without a location writes every species of the default
  acoustic model (11,560 for BirdNET 3.0), in the model's order. v2.4.0 wrote the few
  species the geo model predicted for latitude and longitude -1.
* Lists for a location come from the 3.0 geolocation model and include species the
  acoustic model does not have, such as insects. They are sorted alphabetically; v2.4.0
  used the acoustic model's labels and sorted them by occurrence frequency.

**Consequence.** Scripts that pass -1 get a filter for the wrong place or an error.
Using a location list as ``--slist`` logs the species the model lacks as skipped, and
fails with ``--strict``. Lists without a location are much longer, and the first such
run downloads the acoustic model.

**What to do.** Omit ``--lat``, ``--lon`` and ``--week`` rather than passing -1, and
drop ``--strict`` for lists made by ``birdnet_analyzer.species``.


Top-N results keep the confidence threshold
-------------------------------------------

**What changed.** ``--top_n`` keeps at most N detections per segment among those above
``--min_conf``. v2.4.0 ignored the threshold when ``--top_n`` was set.

**Consequence.** The same command returns fewer detections than in v2.4.0.

**What to do.** Pass a low threshold, such as ``--min_conf 0.00001``, to get the top N
regardless of their score.


Results are written to combined files by default
------------------------------------------------

**What changed.** An analysis writes one combined result file per result type, such as
``BirdNET_SelectionTable.txt``. v2.4.0 wrote one file per audio file and added
combined files only with ``--combine_results``.

**Consequence.** Scripts that expect a ``<audio name>.BirdNET.selection.table.txt``
(or another per-file result) next to each recording find nothing.

**What to do.** Pass ``--split_tables`` for one result file per audio file, which
replaces the combined files. Unlike v2.4.0, it writes no file for an audio file without
detections.


Result and parameters file formats
----------------------------------

**What changed.**

* Raven selection tables add a ``Scientific Name`` column and order the columns
  differently. They have no ``nocall`` row for a file without detections, and no
  ``BirdNET_SelectionTable.list.txt`` is written.
* Audacity labels read ``Scientific name_Common name`` instead of
  ``Scientific name, Common name``.
* The analysis parameters are written to ``birdnet.analyze-params.csv`` instead of
  ``BirdNET_analysis_params.csv``. ``analyze()`` in Python writes it only with
  ``save_params=True``; v2.4.0 always wrote it.
* The parameters of a trained classifier go to ``<classifier>.birdnet.train-params.csv``
  instead of ``<classifier>_Params.csv``.

**Consequence.** Scripts that read these files by column position, label format or
file name break.

**What to do.** Read the columns by name. ``--load_params`` reads both the new and the
v2.4.0 parameters files.


Kaleidoscope output has seven columns
-------------------------------------

**What changed.** ``--rtype kaleidoscope`` writes ``INDIR``, ``FOLDER``, ``IN FILE``,
``OFFSET``, ``DURATION``, ``TOP1MATCH`` and ``TOP1DIST``. v2.4.0 wrote 13 columns: the
scientific and common name, ``confidence``, ``lat``, ``lon``, ``week``, ``overlap`` and
``sensitivity``.

**Consequence.** Scripts that read the old columns break. ``TOP1MATCH`` holds the
common name and ``TOP1DIST`` the confidence, where higher is better, unlike
Kaleidoscope's own distance. ``birdnet_analyzer.segments`` still reads v2.4.0
Kaleidoscope files.

**What to do.** For the scientific name or the analysis settings, use ``--rtype csv``
with ``--additional_columns``.


Custom classifiers must be tflite files
---------------------------------------

**What changed.** ``--classifier`` takes a ``.tflite`` file only; v2.4.0 also loaded a
Raven model folder. The labels are read from ``<classifier>_Labels.txt`` or
``--cc_species_list``; v2.4.0 also found ``<name>_Labels.txt`` next to an older
BirdNET ``<name>_Model_FP32.tflite``.

**Consequence.** A Raven model folder fails to load, and an older BirdNET model needs
its labels file named explicitly.

**What to do.** Use the tflite export of a classifier, and pass ``--cc_species_list``
for older BirdNET models.


Search needs the audio root and a new database
----------------------------------------------

**What changed.** ``birdnet_analyzer.search`` requires ``--audio_root``, the folder the
audio paths in the database are relative to, and ``search()`` takes it as its third
argument, before ``queryfile``. Embedding databases now use the perch-hoplite 1.x
format; v2.4.0 wrote them with perch-hoplite 0.x.

**Consequence.** v2.4.0 search command lines and positional calls of ``search()`` fail.
Databases written by v2.4.0 are in the older format.

**What to do.** Pass ``--audio_root`` (the parent of the input that was embedded), and
extract the embeddings again into a new database.


Evaluation needs a confidence column and scores every annotated recording
-------------------------------------------------------------------------

**What changed.**

* A prediction table without the mapped confidence column, or with an empty one, is an
  error; v2.4.0 scored its predictions as confidence 0.
* Without ``--recording_duration``, a recording that has annotations but no predictions
  is scored, its length taken from the annotations; v2.4.0 dropped it.
* A recording that has predictions but no annotation file is dropped; v2.4.0 counted
  its predictions as false positives.
* The default ``--metrics`` are ``auroc precision recall f1 ap`` instead of
  ``accuracy precision recall``.

**Consequence.** An evaluation with a wrong column mapping fails instead of reporting
meaningless metrics, and the same evaluation reports other metrics and other values
than in v2.4.0.

**What to do.** Map the confidence column with ``--columns_predictions``. Pass
``--recording_duration`` if every recording has the same length,
``--score_unannotated_as_empty`` to count unannotated recordings, and ``--metrics`` for
the v2.4.0 metrics.


Removed options and parameters
------------------------------

**What changed.** These v2.4.0 command-line options no longer exist, so command lines
that use them fail with "unrecognized arguments". The Python functions lost the
matching parameters.

* ``analyze``: ``--threads`` (use ``--n_workers``), ``--combine_results`` (see above)
  and ``--skip_existing_results`` (an interrupted directory analysis resumes instead; a
  finished one runs again). ``analyze()`` also lost ``use_perch``; pass
  ``model="perch"``.
* ``embeddings``: ``--threads`` (use ``--n_workers``).
* ``species``: ``--sortby``.
* ``train``: ``--model_format`` (use ``--model_formats``, ``tflite raven`` for
  ``both``), ``--cache_mode`` and ``--cache_file`` (pass a cache file as ``INPUT`` to
  train from it, and ``--save_cache_to`` to write one), and
  ``--autotune_executions_per_trial`` (``--autotune_n_splits`` with
  ``--autotune_n_repeats`` repeats each trial).

The HTTP server and client, ``birdnet_analyzer.network.server`` and ``.client``, were
removed.

**What to do.** Update scripts as listed.


Other changes
-------------

* **Perch** is not available on macOS — it needs a TensorFlow build that is not yet
  published for that platform.
* **Search** with a query longer than one segment (``--crop_mode segments``) ranks the
  results by their mean score over all query segments. v2.4.0 counted a result only for
  the query segments whose own top results contained it, so its rankings differ.
