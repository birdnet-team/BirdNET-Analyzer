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
location-based species lists use the 3.0 geolocation model. v2.4 was the
previous default for both.

**Consequence.** Results are not directly comparable to v2.4 output: 3.0 is a different
model with different confidence scores, a larger species set (11,725 labels) and an
updated taxonomy. Do not mix or compare 3.0 result files with v2.4 ones.

**What to do.** To keep the previous acoustic model, select the 2.4 model explicitly:

.. code-block:: bash

   birdnet-analyze INPUT --birdnet 2.4

In the GUI, choose *BirdNET 2.4* under model selection. The 2.4 model remains fully
supported, and a location (``--lat`` and ``--lon``) limits it, and a custom classifier
trained on it, with the 2.4 geolocation model, which has the same species.
``birdnet_analyzer.species`` writes its lists with the 3.0 geolocation model either way.


Inference runs on ONNX, not TensorFlow, on the default path
-----------------------------------------------------------

**What changed.** The default 3.0 analysis and geolocation paths run on
``onnxruntime`` instead of TensorFlow, which is now a required dependency.

**Consequence.**

* Default analyses are faster on CPU and no longer load TensorFlow.
* TensorFlow is still required — and still installed — for the 2.4 model, the Perch
  model, custom classifiers, and training.
* The Docker image is substantially larger: it bakes the ~345 MB 3.0 ONNX model
  (the 2.4 TFLite model was ~120 MB).

**What to do.** Reinstall the package so ``onnxruntime`` is pulled in (your usual
``pip install`` step). The 2.4, Perch, custom-classifier and training paths still run
on TensorFlow.


Scores of the BirdNET 2.4 model
-------------------------------

**What changed.** The birdnet library resamples and band-pass filters the audio segment
by segment. v2.4.0 did both on the whole file, with another resampler.

**Consequence.** With ``--birdnet 2.4``, recordings at 44.1 kHz and above give the
detections of v2.4.0, with confidences that differ by less than 0.001; so do stereo
files, flac and mp3, an overlap, a sensitivity, a location and a species list. The
scores differ

* for recordings below 44.1 kHz. On twenty field recordings at 32 kHz, about one
  detection in six above a confidence of 0.25 was found by only one of the two versions,
  the confidences of the others differed by 0.02 in the median and by up to 0.3, and the
  most confident species of a recording was the same in 19 of 20;
* with ``--audio_speed`` other than 1, to a similar degree;
* with ``--fmin``, by up to 0.1 for single detections, and slightly with ``--fmax``.

A custom classifier trained with v2.4.0 gives the scores of v2.4.0 under the same
conditions.

**What to do.** Do not continue a dataset analysed with v2.4.0 under one of these
conditions with this version: analyse all of it with one version.


The installers contain BirdNET 3.0, not 2.4
-------------------------------------------

**What changed.** The installers contain the default BirdNET 3.0 model and its
geolocation model. The BirdNET 2.4 model, which the installer of v2.4.0 contained, is
downloaded the first time it is used (about 80 MB), like every other model.

**Consequence.** An analysis with the default model needs no internet connection.
The first analysis with BirdNET 2.4 does, unless the model is there already: the
settings tab lists the models and whether they are on the computer, and a line below
the model choice says when the first use is going to download one. A location filter
with BirdNET 2.4 uses the geolocation model of 2.4, a download of its own (about
46 MB), which the location settings point out the same way.

**What to do.** Nothing, with an internet connection. When the Windows installer
replaces an installation of v2.4.0, it keeps that installation's BirdNET 2.4 model,
so analyses with BirdNET 2.4 need no download. The geolocation model of 2.4 is not
among the files it can keep: the first analysis with BirdNET 2.4 and a location
still downloads it. For computers without internet access see :ref:`installation`.


Custom species lists and the 3.0 taxonomy
-----------------------------------------

**What changed.** The 3.0 model uses an updated taxonomy, so labels from a v2.4-era
species list may no longer match verbatim — for example ``Accipiter cooperii`` became
``Astur cooperii``, and several common names changed (``Rock Pigeon`` → ``Rock Dove``,
``Common Raven`` → ``Northern Raven``, ``European Starling`` → ``Common Starling``).
Several dozen species were also split, and the scientific name stayed with one of the
new species: ``Larus argentatus`` is now the European Herring Gull and the American
Herring Gull is ``Larus smithsonianus``; ``Setophaga petechia`` is now the Mangrove
Warbler and the American Yellow Warbler is ``Setophaga aestiva``.

A species list passed with ``--slist`` is now **reconciled** to the loaded model
instead of having to match it exactly: each entry is matched by its exact label, then
its scientific name, then its common name in the label language or in English. This
carries most legacy lists over automatically.

**Consequence.**

* Species that still cannot be matched (genuinely absent from the model, or a typo)
  are **skipped with a warning**; v2.4.0 skipped them silently.
* If **none** of the list's species can be matched, the analysis **errors**; v2.4.0
  produced no detections.
* ``--slist`` together with ``--lat`` and ``--lon`` is an error; v2.4.0 ignored the
  list.
* An entry for a species that was split keeps only the species that has its
  scientific name now, **without a warning** and also with ``--strict``:
  ``Larus argentatus_Herring Gull`` no longer covers the American Herring Gull, nor
  ``Setophaga petechia_Yellow Warbler`` the American Yellow Warbler. Barn Owl,
  Warbling Vireo and Cory's Shearwater are other examples.

**What to do.** Most legacy lists work unchanged. Look through a list for species that
were split and add the ones you mean. To require that every entry match — turning any
unmatched species into a hard error — use the new ``--strict`` flag:

.. code-block:: bash

   birdnet-analyze INPUT --slist my_list.txt --strict

In the GUI, selecting a species-list file shows a warning if it contains species the
chosen model does not have (only once that model has been downloaded).


Label languages
---------------

**What changed.** The ``--locale`` codes depend on the model version. Compared with
v2.4.0:

* ``el``, ``he``, ``in``, ``is``, ``ml`` and ``pt_BR`` are no longer accepted.
  English is ``en_us``, the new default; ``en`` is still understood as English, and
  upper and lower case make no difference (``en_UK`` is ``en_uk``).
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


Training chooses its learning rate for the base model
-----------------------------------------------------

**What changed.** Without ``--learning_rate``, training uses a learning rate chosen for
the base model: 0.1 for BirdNET 3.0 and 0.01 for BirdNET 2.4. v2.4.0 used 0.0001, which
leaves classifiers under-trained, on the 3.0 base badly. The train tab no longer offers
the hyperparameters, autotune or the "first" crop mode and always trains with the
defaults. All of them remain command-line options and arguments of ``train()``.

**Consequence.** The same command trains a different classifier than v2.4.0, usually a
much better one, with different confidence scores, so a threshold chosen for an older
classifier no longer fits. Most trainings stop early, well before ``--epochs``. Presets
and parameters files of earlier versions load in the train tab without their
hyperparameters, with "first" as "center". ``--load_params`` still restores the
hyperparameters of a run and its base model, so a v2.4.0 run comes back with its
learning rate of 0.0001; a learning rate recorded for one base model is not used for
training on the other.

**What to do.** Check the ``--min_conf`` used with a retrained classifier. An explicit
``--learning_rate`` overrides both the default and a restored one, and
``--learning_rate 0.0001`` keeps the old value. With ``--hidden_units``, also pass a
lower learning rate, such as 0.003.


Locations and species lists
---------------------------

**What changed.**

* ``-1`` no longer means "not set". ``--lat -1 --lon -1`` is a location in the Gulf of
  Guinea, and ``--week -1`` with a location is an error. Leave an option out instead.
* ``analyze`` and ``species`` reject only one of ``--lat`` and ``--lon``; v2.4.0 used
  -1 for the missing one.
* ``birdnet_analyzer.species`` without a location writes every label of the default
  acoustic model (11,725 for BirdNET 3.0: 11,709 species and 16 sound classes), in the
  model's order. v2.4.0 wrote the few species the geo model predicted for latitude and
  longitude -1.
* Lists for a location come from the 3.0 geolocation model and include species the
  acoustic model does not have, such as insects. They are sorted alphabetically; v2.4.0
  used the acoustic model's labels and sorted them by occurrence frequency.

**Consequence.** Scripts that pass -1 get a filter for the wrong place, which finds
next to nothing, or an error. ``--lat -1 --lon -1`` logs a warning that says so.
Using a location list as ``--slist`` logs the species the model lacks as skipped, and
fails with ``--strict``. A species with another scientific name in the acoustic model
is found by its common name, so it is skipped too when the list is neither in English
nor in the language of the analysis, or when the 2.4 model also names it differently.
Lists without a location are much longer, and the first such run downloads the acoustic
model.

**What to do.** Omit ``--lat``, ``--lon`` and ``--week`` rather than passing -1, and
drop ``--strict`` for lists made by ``birdnet_analyzer.species``. Write a list in
English or in the ``--locale`` of the analysis. With ``--birdnet 2.4``, pass ``--lat``
and ``--lon`` to the analysis instead of a list.


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
replaces the combined files. As in v2.4.0, an audio file without detections still gets
its files, which hold only the header.

Analyse into a new output folder when the old one holds the per-file results of an
earlier analysis: they stay where they are, and the evaluation reads the per-file
tables of a folder in place of a combined table beside them.


Result and parameters file formats
----------------------------------

**What changed.**

* Raven selection tables add a ``Scientific Name`` column and order the columns
  differently. They have no ``nocall`` row for a file without detections, and no
  ``BirdNET_SelectionTable.list.txt`` is written. As before, the times of a combined
  table run on across all analysed files in path order, including those without
  detections. The ``Species Code`` is found by the scientific name, in every label
  language, and by the English name where the 3.0 model uses a newer scientific name.
  A species the eBird list of the 2.4 model does not have, as close to half of the 3.0
  model's, carries its label there.
* Confidences are written with all their digits (``0.81405586``); v2.4.0 rounded them
  to four decimals.
* With ``--additional_columns``, ``lat``, ``lon`` and ``week`` are empty without a
  location, where v2.4.0 wrote ``-1``, and ``model`` is the file name of the model in
  the model directory.
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
v2.4.0 parameters files. An analysis file of v2.4.0 held only the overlap, the bandpass,
the merging, the audio speed and the classifier; it is repeated with BirdNET 2.4, or
with Perch if it was a Perch analysis, and with the defaults for everything else.


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
format; v2.4.0 wrote them with perch-hoplite 0.x. A new database holds the embeddings
of BirdNET 3.0, which have 1280 values per segment; v2.4.0 extracted them with BirdNET
2.4, with 1024 values.

**Consequence.** v2.4.0 search command lines and positional calls of ``search()`` fail.
Databases written by v2.4.0 are in the older format. Embeddings and search scores are
not comparable with those of v2.4.0.

**What to do.** Pass ``--audio_root`` (the parent of the input that was embedded), and
extract the embeddings again into a new database. ``birdnet_analyzer.embeddings
--birdnet 2.4`` fills a new database with BirdNET 2.4 embeddings; the GUI always uses
BirdNET 3.0 for a new database.


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


Evaluation reads tables differently
-----------------------------------

**What changed.**

* The columns are detected from the tables' headers. ``--columns_annotations`` and
  ``--columns_predictions`` name only the columns that differ; v2.4.0 took every column
  it was not given by its literal field name. A column named ``Duration`` is no longer
  read as the recording length (BirdNET's ``File Duration (s)`` is).
* The annotation and prediction paths are searched with their subfolders, for ``.txt``
  and ``.csv`` tables; v2.4.0 read the ``.txt`` files directly in the folder. A recording
  name in two subfolders, and the tables of one recording in two subfolders, such as two
  analyses, are errors, and so is a table read on both sides.
* Two named files (``--annotation_path`` and ``--prediction_path`` both naming a file)
  are compared as one recording when each holds one recording, whatever their names;
  v2.4.0 paired them only if their names matched. When the prediction file is a
  combined table of several recordings, only the recording named like the annotation
  file is scored, as with folders; v2.4.0 also scored the others, as recordings without
  calls.
* Tables of which no recording pairs are an error; v2.4.0 scored their annotations as
  missed and their predictions as false positives.
* ``process_data()`` returns an ``EvaluationResult`` instead of the 4-tuple
  ``(metrics_df, pa, predictions, labels)``; its first four fields are those values.
  ``read_and_concatenate_files_in_directory``, ``extract_recording_filename`` and
  ``extract_recording_filename_from_filename`` were removed from
  ``birdnet_analyzer.evaluation.preprocessing.utils``; use ``read_tables`` and
  ``recording_key``.
* The GUI's **Download results table** has one row per class and the overall row last,
  with the metrics as columns; v2.4.0 had the metrics as rows. The command line's
  ``results_table.csv`` keeps the metrics as rows.

**Consequence.** Command lines and scripts that relied on the literal column names, on
the folder holding only top-level ``.txt`` tables, or on the removed functions read other
tables or fail, and file pairs of a combined table score other values than in v2.4.0.

**What to do.** Name only the columns that are not detected, and select folders that hold
one analysis or one set of annotations each. For a recording length column named
``Duration``, pass ``--columns_annotations '{"Duration": "Duration"}'`` (or
``--columns_predictions``), or ``--recording_duration``.


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
* ``train``: ``--model_format both`` (use ``--model_formats tflite raven``; with one
  format the old spelling still works), ``--cache_mode`` and ``--cache_file`` (pass a cache file as ``INPUT`` to
  train from it, and ``--save_cache_to`` to write one), and
  ``--autotune_executions_per_trial`` (``--autotune_n_splits`` with
  ``--autotune_n_repeats`` repeats each trial).

The HTTP server and client, ``birdnet_analyzer.network.server`` and ``.client``, were
removed. So were the modules the analysis was built from: ``birdnet_analyzer.analyze.utils``,
``birdnet_analyzer.embeddings.utils``, the ``predict``, ``embeddings`` and ``load_model``
functions of ``birdnet_analyzer.model``, and the run-time settings of
``birdnet_analyzer.config`` such as ``cfg.MIN_CONFIDENCE``. ``analyze()`` returns the
result of the birdnet library instead of ``None``.

**What to do.** Update scripts as listed.


Other changes
-------------

* **Search** with a query longer than one segment (``--crop_mode segments``) ranks the
  results by their mean score over all query segments. v2.4.0 counted a result only for
  the query segments whose own top results contained it, so its rankings differ.
* **Merged detections** (``--merge_consecutive``) of more than three segments have the
  mean confidence of all of them; v2.4.0 took the mean of the three highest.
* **The GUI of a pip or source installation** keeps its language, theme and last used
  folders in the user's application data folder, as the installed application always
  did; v2.4.0 kept them in the package folder, so they start from the defaults once.
* **The GUI's worker count** is automatic by default: the birdnet library starts as
  many worker processes as fit the available memory, at most one per CPU core, where
  v2.4.0 started one per logical core. A count set in an earlier version is ignored
  until "Automatic worker count" is unchecked.
* **m4a and wma files** are no longer offered, neither by the file dialog nor by the
  folder scans of the analysis, training, segments and review tabs. The birdnet
  library reads the formats libsndfile supports (wav, flac, mp3, ogg, aiff, ...);
  v2.4.0 decoded the others through audioread, which needs a system decoder (CoreAudio
  on macOS, ffmpeg elsewhere) that the installers never shipped. Convert such files to
  wav or flac first.
* **Segment folders** are named after the detection's common name with the characters a
  folder name cannot hold (``< > : " / \ | ? *``) replaced by ``-`` and a trailing dot or
  space dropped. v2.4.0 used the name as it was, which nested the folders of a name with
  ``/`` (``Eurasian/Green-winged Teal``) and failed on Windows for a name with ``?``.
  Segments extracted again into an existing output folder land in the new folder.
