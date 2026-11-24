Evaluation Tool
===============

The **Evaluation** tab compares the predictions of an analysis, made with BirdNET, Perch or a custom classifier, with your own annotations of the same recordings.
It shows how well the model works on *your* recordings and helps you choose a confidence threshold for them.
The same evaluation runs on the command line as :ref:`birdnet_analyzer.evaluation <cli-evaluation>`.

What the tab measures
---------------------

The evaluation does not match individual calls.
It cuts every recording into consecutive **windows**, starting at 0 s.
Their length, the **Window length**, is that of the segments the model scores: 3 s for BirdNET, 5 s for Perch.
Each window is then judged once per class:

* It is **annotated** for a class if an annotation of the class overlaps it by at least the **Minimum overlap** (0.5 s by default).
* Its **score** for a class is the highest confidence among the predictions of the class that overlap it by at least the minimum overlap, or 0 if there is none.
* It is **detected** for a class if that score is at or above the **Confidence threshold**.

Together, the two give the four outcomes every metric is computed from:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Window
     - detected
     - not detected
   * - annotated
     - correct detection (true positive)
     - missed call (false negative)
   * - not annotated
     - false positive
     - correct rejection (true negative)

The scores therefore count **windows, not calls**: a song that lasts 12 s counts in four or five windows, and two calls in the same window count once.
The **Annotated windows** column of the results shows how many windows each class has, the number of positives its scores rest on.

For example, take a 12 s recording, 3 s windows, the default minimum overlap of 0.5 s and a confidence threshold of 0.5.
A Song Thrush is annotated from 2.0 s to 7.4 s, and the predictions contain Song Thrush at 3–6 s with a confidence of 0.82, at 6–9 s with 0.35 and at 9–12 s with 0.61:

.. list-table::
   :header-rows: 1

   * - Window
     - Annotation overlaps it by
     - Score
     - Outcome for Song Thrush
   * - 0–3 s
     - 1.0 s: annotated
     - 0 (no prediction)
     - missed call
   * - 3–6 s
     - 3.0 s: annotated
     - 0.82: detected
     - correct detection
   * - 6–9 s
     - 1.4 s: annotated
     - 0.35: below the threshold
     - missed call
   * - 9–12 s
     - not at all
     - 0.61: detected
     - false positive

In this recording, Song Thrush has 3 annotated windows, a recall of 1/3 and a precision of 1/2.

.. note::

   The exact rule: an annotation or prediction counts for a window if it ends at least the minimum overlap after the window starts and begins at least the minimum overlap before the window ends.
   A call shorter than the minimum overlap therefore still counts for a window it lies well inside, but one near a window border may count for no window:
   with the default 0.5 s, a 0.2 s call from 2.9 s to 3.1 s counts neither for the window 0–3 s nor for 3–6 s.
   Lower the minimum overlap if you evaluate very short calls.

Preparing your data
-------------------

You need two directories: one with your annotations and one with the results of an analysis of the same recordings.

Annotations
###########

* Save **one table per recording**, such as a Raven selection table. Each row is one call with its start time, end time and class; other columns are ignored. The column names are detected automatically (see :ref:`eval-columns`).
* **Name each table after its audio file**: ``site1_0530.wav`` → ``site1_0530.txt``, or Raven's default ``site1_0530.Table.1.selections.txt`` (see :ref:`eval-pairing`).
* **Annotate every call of the classes you evaluate**, in every annotated recording. A window without an annotation counts as "no call", so a call you skipped turns a correct detection into a false positive. Classes you do not evaluate need no annotations.
* **A recording without calls needs a table too**: one with only the header row, or an empty file. The recording then counts as checked and without calls, and every detection in it is a false positive. A recording without a table is not scored, unless you turn on **Score recordings without annotation table as empty**.
* The tables are ``.txt`` or ``.csv`` files, separated by tabs, semicolons or commas. Numbers need a decimal point (``1.5``, not ``1,5``).

Predictions
###########

* Analyze the same recordings in the **Batch analysis** tab or with :ref:`birdnet_analyzer.analyze <cli-docs>`, and select its output directory as the prediction directory.
* The **Raven selection table** (the default) and the **CSV** result types both work, as one combined table for all recordings (the default) or as one table per recording (**Split tables by input file**). Audacity labels have no header row and Kaleidoscope tables no end time, so neither can be evaluated; Parquet files are not read.
* **Analyze with a low minimum confidence.** The result tables only contain predictions at or above the analysis' **Minimum confidence**, and every other window scores 0. Thresholds below it all give the same results, and AUROC and AP, which rank all windows by their score, depend on it. The **Minimum confidence** of the Batch analysis tab goes down to 0.05; on the command line, analyze with ``--min_conf 0.01``.
* Analyze with the settings you want to evaluate, such as the model, the species list or location filter and the sensitivity: they all change the predictions, and a species missing from the species list is never predicted.

.. _eval-pairing:

How tables are paired
#####################

An annotation table and a prediction table belong together if they have the same **recording name**: the file name without its extension and without the suffixes BirdNET and Raven add, compared ignoring case.
All of these belong to the recording ``site1_0530.wav``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File name, or value in the Recording column
     - Written by
   * - ``site1_0530.txt``, ``site1_0530.csv``
     - you, named after the audio file
   * - ``site1_0530.Table.1.selections.txt``
     - Raven, saving a selection table
   * - ``site1_0530.wav.txt``
     - a tool that keeps the audio extension
   * - ``site1_0530.BirdNET.selection.table.txt``, ``site1_0530.BirdNET.results.csv``
     - BirdNET, with **Split tables by input file**
   * - ``D:\audio\site1_0530.WAV`` in the ``Begin Path`` or ``File`` column
     - BirdNET, in every result table
   * - ``SITE1_0530.txt``
     - anyone: case does not matter

Dots inside a name are kept, so ``2024.05.30_dawn.txt`` and ``2024.05.31_dawn.txt`` stay two recordings.

* If a **Recording** column is selected under **Annotation columns** or **Prediction columns**, each row belongs to the recording named in that column, and a row with an empty value to its table's file name. This is how a combined table is divided into recordings. The column is detected automatically from names such as ``Begin File``, ``Begin Path`` or ``File``; clear the dropdown to pair by file names only.
* BirdNET's combined Raven table counts ``Begin Time (s)`` from the start of the first recording. Its ``File Offset (s)`` column holds the time within each recording, and the evaluation uses it, so times always count from the start of their recording. The same works for Raven selection tables of a file sequence that have the ``Begin File`` and ``File Offset (s)`` columns.
* Several tables of one recording in the same folder, such as ``Table.1`` and ``Table.2``, are combined.

Subfolders and unique names
###########################

Both directories are searched with all their subfolders for ``.txt`` and ``.csv`` files. Left out are:

* hidden files and folders (names starting with a dot) and parameter files (names ending in ``params.csv``, such as ``birdnet.analyze-params.csv``);
* BirdNET's combined selection table and Audacity labels in a folder with per-recording BirdNET selection tables, and its combined CSV table in a folder with per-recording BirdNET CSV tables. The summary lists them as ignored;
* files without the selected columns, such as a ``README.txt``. The summary lists them as skipped.

Because tables are paired by name, a recording name may occur in only one folder of each directory.
Recorders that name their files by date and time give the same names at every site, so tables for ``20240530_053000`` in both ``site1`` and ``site2`` are rejected.
Evaluate one site or recorder at a time, or give the recordings unique names.

For the same reason, all tables of a recording must be in one folder.
A prediction directory that holds several analyses of the same recordings in its subfolders, such as ``results/v2.4`` and ``results/v3.0``, is rejected rather than pooled: select the folder of one analysis.
The same applies to two sets of annotations of a recording in two folders.

The two directories must hold different tables: a table that would be read as an annotation table and as a prediction table, as when the same folder is selected twice or the annotation directory contains the prediction directory, is rejected.

.. _eval-class-names:

Class names
###########

Classes are matched by name, ignoring case and extra spaces: the annotation ``great tit`` and the prediction ``Great Tit`` are the same class.
The results use the annotations' spelling.

BirdNET's result tables name each prediction by its common name, in the language set as **Locale** for the analysis, and by its scientific name; Raven tables also by a species code.
Perch names its predictions by their scientific names only.
If your annotations use other names, the summary lists the annotated classes it does not find in the predictions. There are three remedies:

1. **The prediction class column switches automatically.** When you select a directory and no class mapping is loaded, the tab looks for the prediction column that holds the most annotated class names (``Common Name``, ``Scientific Name``, ``Species Code`` and the like) and uses it. You can also choose it yourself: **Prediction columns** → **Class**.
2. **Use these name matches.** For an annotated class that does not occur in the predictions, the tab looks for a predicted class without annotations that occurs in at least a third of the class's annotated windows and scores there, on average, at least four times as high as in the other windows. That is usually the same species under another name, such as ``Common Chaffinch ← Eurasian Chaffinch``, but not always: check each pair in **Name matches** below the summary and clear any that is not the same species. Then click **Use these name matches**: the checked pairs are added to the class mapping under **Class mapping (optional)**, or written into a new one there.
3. **A class mapping file.** A JSON file that renames predicted classes to the names in your annotations, predicted name first:

   .. code-block:: json

      {
          "Eurasian Chaffinch": "Common Chaffinch",
          "Common Blackbird": "Eurasian Blackbird"
      }

   Write each predicted name exactly as it appears in the selected class column of the prediction tables, including its case. **Download template** under **Class mapping (optional)** saves a file to start from; load yours with **Upload mapping file**. While a mapping is loaded, the class column does not switch automatically.

Step by step
------------

1. Prepare the two directories as described above. The **Info** box at the top of the tab sums up the steps.
2. Click **Select annotation directory** and choose the folder with your annotation tables. A first summary line shows what was read, and **Annotation columns** appears.
3. Click **Select prediction directory** and choose the output folder of the analysis. If it holds the analysis' ``birdnet.analyze-params.csv``, the **Window length (s)** is set to the analysis' segment length and the **Confidence threshold** is raised to its minimum confidence (see :ref:`eval-parameters`). You can select the directories in either order.
4. Read the **summary** below the directories (see below) and fix what it marks with ⚠️. While the tables are read, the button shows **Reading the tables…** and cannot be clicked.
5. Optionally, check the columns, load a class mapping, narrow down the classes or recordings, and adjust the parameters and metrics.
6. Click **Calculate metrics**. The results appear below the button.

Reading the summary
###################

The summary is updated when you select a directory or change a column, the class mapping, **Window length (s)**, **Recording duration (s)**, **Minimum overlap (s)** or **Score recordings without annotation table as empty**.
If you change, add or remove tables in the directories afterwards, select the directory again to read them: until then, **Calculate metrics** asks you to.
Its first block says what was read, the second whether the two sides fit together, and the third which files were left out.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Line (example)
     - Meaning and what to do
   * - **Annotations** · tables: 8 · annotations: 65 · classes: 10
     - The annotation tables read, their rows and the number of different class names. Check them against your annotation work.
   * - ℹ️ Annotation tables without rows, counted as recordings without calls: 2
     - Tables with only a header row, or empty files: these recordings count as annotated, without calls.
   * - **Predictions** · tables: 1 · detections: 252 · classes: 32
     - The same for the predictions. One table is a combined result table.
   * - Analysis: BirdNET 3.0, 3 s segments, minimum confidence 0.25
     - Read from the analysis' ``birdnet.analyze-params.csv``: the model, its segment length and the minimum confidence of the analysis.
   * - ⚠️ The annotation tables have a "Confidence" column, like prediction tables. …
     - The annotation directory may hold BirdNET's results instead of your annotations; scored as annotations, they only compare BirdNET with itself. Detections you checked and corrected, used as annotations, are fine.
   * - ✅ Recordings with annotations and predictions: all 8
     - Every recording has both an annotation table and predictions.
   * - ⚠️ Recordings with annotations and predictions: 6 of 8
     - Some recordings have only one side; the lines below name them, followed by the pairing rule. If none or few pair, see :ref:`eval-pairing`.
   * - Annotations only, so all their annotations count as missed: site2_0530
     - No predictions belong to this recording. A combined table has no rows for a recording without any detection, so such a recording is listed here and still scored correctly. Otherwise, check that the analysis covered the recording and that the names pair.
   * - Predictions only, not scored: site3_0530
     - No annotation table belongs to this recording, so it is left out. If you checked the recording, give it a table, with only the header row if it has no calls, and select the annotation directory again.
   * - Predictions only, scored as recordings without calls: site3_0530
     - As above, with **Score recordings without annotation table as empty** turned on: every prediction in the recording counts as a false positive.
   * - ✅ Annotated classes found in the predictions (column "Common Name"): all 10
     - Every annotated class occurs in the predictions, in the column named.
   * - ⚠️ Annotated classes found in the predictions (column "Common Name"): 7 of 10. Not found: …
     - The classes named never occur in the predictions under these names. Either the names differ (see :ref:`eval-class-names`) or the model never predicted them; such a class gets a recall of 0.
   * - 💡 Annotated classes that probably occur in the predictions under another name: 3. Check the name matches below.
     - The suggested pairs are listed under **Name matches**, each checked: ``Common Chaffinch ← Eurasian Chaffinch (17/21 windows)`` means that Eurasian Chaffinch was detected in 17 of the 21 windows annotated as Common Chaffinch. Clear any pair that is not the same species, then click **Use these name matches**.
   * - ℹ️ The prediction class column was set to "Scientific Name", which holds the most annotated classes.
     - The automatic switch of the class column. It is also shown under **Prediction columns** → **Class**.
   * - ⚠️ The window length (3 s) differs from the segment length of the analysis (5 s).
     - Set **Window length (s)** to the segment length, unless you have a reason not to.
   * - ℹ️ Skipped files in the prediction directory without the selected columns (1): README.txt
     - Files that lack a selected column or cannot be read as a table. Fine for notes; for a table, check its columns.
   * - ℹ️ Ignored other BirdNET outputs next to the per-recording tables (1): BirdNET_SelectionTable.txt
     - Combined tables or Audacity labels in a folder with per-recording tables. Only the per-recording tables are read.

A problem that prevents reading the tables replaces the summary, marked ⚠️; see :ref:`eval-troubleshooting`.

.. _eval-columns:

Annotation columns and prediction columns
#########################################

**Annotation columns** and **Prediction columns** hold one dropdown per field: **Start time**, **End time** and **Class**, **Confidence** for the predictions, **Recording (optional)** and **Duration (optional)**.
They list the columns found in the tables of the directory and are filled in automatically from the column names below.
The first name in the list that a table has exactly is used, otherwise the first that matches ignoring case.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Column names tried, in this order
   * - Start time
     - ``Begin Time (s)``, ``Start (s)``, ``Start Time``, ``Begin Time``, ``Start``
   * - End time
     - ``End Time (s)``, ``End (s)``, ``End Time``, ``End``
   * - Class (annotations)
     - ``Class``, ``Species``, ``Label``, ``Annotation``, ``Common Name``, ``Scientific Name``, ``Species Code``, ``Sound Type``, ``Call type``
   * - Class (predictions)
     - ``Common Name``, ``Class``, ``Scientific Name``, ``Species Code``, ``Label``, ``Species``
   * - Confidence
     - ``Confidence``, ``Score``
   * - Recording (annotations)
     - ``Begin File``, ``Begin Path``, ``File``, ``Filename``, ``File Name``, ``Recording``
   * - Recording (predictions)
     - ``Begin Path``, ``Begin File``, ``File``, ``Filename``, ``File Name``, ``Recording``
   * - Duration
     - ``File Duration (s)``

Change a dropdown if a field was detected wrongly.
Leave **Recording (optional)** empty to pair tables by their file names, and **Duration (optional)** empty to take a recording's length from its last annotation or prediction.
If no table has a column you selected for a required field, the summary says so and opens the box.

Other optional settings
#######################

* **Class mapping (optional)**: renames predicted classes, see :ref:`eval-class-names`.
* **Select classes and recordings**: under **Select classes**, the annotated classes come first and are selected; with none selected, they are evaluated. You can add a predicted class without annotations, but it has no annotated windows and shows n/a. Under **Select recordings**, choose recordings to evaluate; with none selected, all are evaluated.
* **Select metrics**: the metrics to calculate (see :ref:`eval-metrics`). **Accuracy** is off by default. **Averaging** sets how the overall row combines the classes (see :ref:`eval-results`).

The parameters and metric settings you change are kept for your next session.

.. _eval-parameters:

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 13 65

   * - Parameter
     - Default
     - What it does
   * - **Window length (s)**
     - 3
     - The length of the windows the recordings are cut into. Use the segment length of the model: 3 s for BirdNET, 5 s for Perch. Selecting a prediction directory sets it to the segment length in the analysis' ``birdnet.analyze-params.csv``.
   * - **Recording duration (s)**
     - empty
     - Leave it empty to take each recording's length from the duration column, or else from its last annotation or prediction. A value applies to every recording and must be at least the window length. BirdNET's result tables have no duration column, so without a value the windows after the last annotation or prediction are not scored, and a recording whose two tables are both empty is not scored at all. If all recordings have the same length, enter it.
   * - **Minimum overlap (s)**
     - 0.5
     - How many seconds an annotation or prediction must overlap a window to count for it (see the note in `What the tab measures`_). More than 0, at most the window length.
   * - **Confidence threshold**
     - 0.1
     - Predictions with at least this confidence count as detections. Selecting a prediction directory raises it to the analysis' minimum confidence, since lower thresholds give the same results. Precision, recall, F1 and accuracy depend on it; AUROC and AP do not.
   * - **Per-class plots**
     - off
     - Draws the **Metrics** and **Metrics across thresholds** plots for each selected class instead of the overall scores.
   * - **Score recordings without annotation table as empty**
     - off
     - Recordings with predictions but no annotation table are scored as if they had no calls, so every prediction on them counts as a false positive. Turn it on only if none of these recordings contains a call of an evaluated class. By default, they are not scored.

.. _eval-results:

Reading the results
-------------------

The header line names what was compared, for example:

   **Recordings:** 8 · **windows:** 223 of 3 s · **classes:** 10 · **confidence threshold:** 0.25

that is, the recordings evaluated, the number of windows and their length, the number of selected classes and the threshold.

The table
#########

The table has one row per selected class, the classes with annotations first.
It has a column for each selected metric, rounded to three decimals, and the **Annotated windows** column: the windows annotated with the class, the positives its scores rest on.
Scores of a class with only a few annotated windows are uncertain.

A selected class without annotated windows in the evaluated recordings has no recall, F1, AP or AUROC, and its precision could only count false positives: its row shows **n/a**, and it is left out of the overall row.

The overall row at the bottom combines the classes with annotations, as chosen under **Averaging** in **Select metrics**:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Averaging
     - Overall row
     - How the classes are combined
   * - Macro (unweighted mean)
     - Overall (mean of 10)
     - The mean of the class values: every class counts the same, rare ones as much as common ones.
   * - Micro (pooled)
     - Overall (pooled, 10)
     - The windows of all classes are pooled and scored as if they were one class: common classes dominate.
   * - Weighted (by annotated windows)
     - Overall (weighted mean of 10)
     - The mean of the class values, weighted by their annotated windows.

The number is the count of classes combined; the row's **Annotated windows** are their sum.
Changing **Averaging** after a calculation updates the overall row and the overall plots right away.

Notes
#####

Below the table, notes point out what affects the numbers:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Note (example)
     - Meaning
   * - ⚠️ Recordings with predictions but no annotation table, not scored (1): site3_0530
     - These recordings are left out. With **Score recordings without annotation table as empty**, the note says they were scored as recordings without calls.
   * - ⚠️ Recordings with annotations but no prediction table; all their annotations count as missed (1): site2_0530
     - Check that the analysis covered these recordings; see the summary line "Annotations only".
   * - 📊 Selected classes without annotations in the evaluated recordings, shown as n/a and left out of the overall row (1): Common Cuckoo
     - The n/a rows of the table.
   * - ℹ️ The predictions also contain classes without annotations (classes: 22, detections: 38), e.g. … They are not part of these scores; if your annotations cover every species present, these detections are false positives.
     - Detections of classes you did not annotate. If you annotated every species, they are misidentifications that the per-class scores do not show.
   * - ℹ️ The predictions contain no confidence below 0.25, so any lower threshold gives the same results as 0.25.
     - The threshold is below the analysis' minimum confidence (or, without a parameters file, the lowest confidence in the tables).
   * - ℹ️ AUROC and AP rank all windows by confidence, but the predictions only contain confidences of 0.25 and above; …
     - Shown for a minimum confidence above 0.05. All lower scores count as 0, so AUROC and AP depend on the minimum confidence: compare them only between analyses that used the same one.
   * - ℹ️ The per-class plots show the 25 classes with the most annotated windows.
     - Shown when more than 25 classes are selected.

Plots
#####

The three tabs below the notes each hold a plot, which **Download plot** saves as PNG, WebP or JPG:

* **Metrics**: the overall value of each selected metric as a bar chart, or, with **Per-class plots**, each metric per class as a line chart.
* **Confusion matrix**: one 2 × 2 matrix per class. Its rows are the windows without (Negative) and with (Positive) an annotation of the class, its columns whether the class was detected at the threshold, and each row adds up to 1. The bottom right value is the recall; the top right one is the share of windows without the class in which it was detected.
* **Metrics across thresholds**: the selected metrics that depend on the threshold (precision, recall, F1 and accuracy) for thresholds from 0.05 to 0.95, overall or, with **Per-class plots**, per class. AUROC and AP do not depend on the threshold and are not drawn.

To choose a threshold, look at **Metrics across thresholds**: a higher threshold usually gives a higher precision and a lower recall.
Pick the threshold whose balance suits your study and set it as **Confidence threshold** to see the values of each class.
The best threshold often differs between classes, which **Per-class plots** shows.

Downloads
#########

* **Download results table** saves the table on screen as CSV or TSV (by the file extension you choose): a ``Class`` column, the metrics and ``Annotated windows``, one row per class and the overall row last. The cells shown as n/a are empty.
* **Download data table** saves the windows themselves: one row per window with ``filename``, ``sample_index``, ``start_time`` and ``end_time``, and for each class ``<class>_confidence`` (the window's score) and ``<class>_annotation`` (1 if annotated). It is not limited to the classes and recordings selected under **Select classes and recordings**. Use it to check single windows or for your own statistics.

.. _eval-metrics:

Metrics
-------

* **AUROC** (area under the receiver operating characteristic curve): the probability that a random window with the class scores higher than a random window without it. 0.5 is chance, 1 a perfect ranking. It does not depend on the confidence threshold.

  * Advantage: Summarizes how well the scores separate windows with and without the class over all thresholds, regardless of how common the class is.
  * Disadvantage: Says nothing about a particular threshold. When most windows lack the class, it can be high although the precision at a useful threshold is low. It depends on the analysis' minimum confidence.

* **Precision**: the share of detections that are correct.

  * Advantage: Tells how much of the output is right, i.e. how many detections you would discard after checking them.
  * Disadvantage: Ignores missed calls and depends on the threshold. A class that is never detected gets 0.

* **Recall**: the share of annotated windows that are detected.

  * Advantage: Tells how many calls the model misses at the threshold.
  * Disadvantage: Ignores false positives: a very low threshold reaches a high recall along with many false positives.

* **F1 Score**: the harmonic mean of precision and recall.

  * Advantage: One number that is only high when both precision and recall are.
  * Disadvantage: Depends on the threshold, weighs precision and recall equally, which may not match your goal, and hides which of the two is low.

* **Average Precision (AP)**: summarizes the precision over all recall levels, i.e. the precision–recall curve across all thresholds. It does not depend on the confidence threshold.

  * Advantage: A single, threshold-free number that focuses on the windows with the class, so it stays informative for rare classes.
  * Disadvantage: Noisy for classes with few annotated windows. Its chance level is the share of windows with the class, so the values of common and rare classes are not directly comparable. It depends on the analysis' minimum confidence.

* **Accuracy**: the share of correct decisions, windows detected or left undetected correctly. Off by default.

  * Advantage: Simple to understand.
  * Disadvantage: Misleading for sound recordings: most windows contain no call of a class, so true negatives dominate and accuracy stays high even for a model that detects nothing. A class present in 5 % of the windows reaches an accuracy of 0.95 without a single detection.

.. _eval-troubleshooting:

Troubleshooting
---------------

A problem found while the tables are read replaces the summary, marked ⚠️; one found when you click **Calculate metrics** appears as an error message.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Message (example)
     - Cause
     - What to do
   * - The prediction tables have no column "Confidence". Choose the column for "Confidence" under "Prediction columns". Columns found: Begin Time (s), End Time (s), Species
     - No table in the directory has the column selected for a required field. Here, the annotation directory was selected as the prediction directory.
     - Check that each button got the right directory. Otherwise, choose the column in the box that opened.
   * - … The files seem to have no header row: their first line holds values, not column names.
     - The tables have no header row, such as Audacity labels.
     - Add a header row, or analyze with the Raven selection table or CSV result type.
   * - site1_0530.txt, row 12: "1,5" in the column "End Time (s)" is not a number. Numbers need a decimal point, not a comma.
     - A time, confidence or duration that is not a number. With the second sentence: a decimal comma, as saved by spreadsheets in some languages.
     - Correct the value (row 1 is the first row below the header), or save the table with decimal points. If the column is the wrong one, choose another.
   * - Two prediction tables belong to recordings named "20240530_053000" (site1, site2). Tables are paired by recording name, so the names must be unique: …
     - The same recording name occurs in two folders of the directory, typical for recorders that name files by date and time.
     - Evaluate one site or recorder at a time, or give the recordings unique names (see `Subfolders and unique names`_).
   * - The prediction tables of the recording "site1_0530" are in two folders (v2.4, v3.0), for example from two analyses, so they would be pooled. …
     - The directory holds several analyses of the same recordings in subfolders. For annotations, the message says the same about two sets of annotations.
     - Select the folder of one analysis, or keep each recording's tables in one folder.
   * - The same tables were selected as annotations and as predictions, e.g. site1_0530.BirdNET.selection.table.txt. …
     - The same folder was selected twice, or the annotation directory contains the prediction directory.
     - Select the folder with your annotations and the output folder of the analysis, two separate folders.
   * - The tables changed since they were read. Select the annotation or prediction directory again to read them anew.
     - Tables in a directory were changed, added or removed after it was selected.
     - Select the directory again.
   * - None of the annotation tables could be paired with a prediction table, so there is nothing to compare. Tables are paired by recording name: …
     - No recording name occurs in both directories.
     - Compare the names the summary lists under "Annotations only" and "Predictions only", and name the annotation tables after the audio files. A combined prediction table needs its **Recording (optional)** column, such as ``Begin Path``.
   * - **Annotations** · no tables (.txt or .csv) found in this directory or its subdirectories.
     - Apart from hidden and parameter files, the directory holds no ``.txt`` or ``.csv`` files that can be read as tables, for example because the tables are spreadsheets.
     - Select the directory with the tables, or save them as ``.txt`` or ``.csv``.
   * - Enter a window length in seconds, e.g. 3 for BirdNET or 5 for Perch.
     - **Window length (s)** is empty, 0 or negative.
     - Enter the segment length of the model.
   * - The minimum overlap must be more than 0 s and at most the window length.
     - **Minimum overlap (s)** is 0, negative, empty or longer than the window.
     - Enter a value such as 0.5.
   * - Enter the recording duration in seconds, at least the window length, or leave it empty.
     - **Recording duration (s)** is not a number or shorter than a window.
     - Correct it, or leave it empty.
   * - The class mapping file cannot be used. It must be a JSON object that maps predicted class names to annotation class names; see the template.
     - The file is not valid JSON, for example because of a comma after the last entry, or it is not an object of names.
     - Compare it with the template from **Download template**.
   * - Select at least one metric.
     - No metric is selected under **Select metrics**.
     - Select one.

Results that look wrong:

* **A class has a precision, recall and F1 of 0 and an AUROC of 0.5**: it never occurs in the predictions under this name. Check the class line of the summary and :ref:`eval-class-names`.
* **A lower threshold changes nothing**: the predictions contain no lower confidences. Analyze with a lower minimum confidence.
* **Many false positives** where the model seems right: calls that were not annotated, or **Score recordings without annotation table as empty** turned on for recordings that do have calls.
* **Unexpected misses**: compare **Window length (s)** with the segment length of the analysis; the summary warns when they differ.
