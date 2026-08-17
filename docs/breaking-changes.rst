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

**What changed.** Analyses use the BirdNET 3.0 acoustic model, and location-based
species lists use the 3.0 geolocation model, by default. v2.4 was the previous default.

**Consequence.** Results are not directly comparable to v2.4 output: 3.0 is a different
model with different confidence scores, a larger species set (11,560 labels) and an
updated taxonomy. Do not mix or compare 3.0 result files with v2.4 ones.

**What to do.** To keep the previous behaviour, select the 2.4 model explicitly:

.. code-block:: bash

   birdnet-analyze INPUT --birdnet 2.4

In the GUI, choose *BirdNET 2.4* under model selection. The 2.4 model remains fully
supported.


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
  are **skipped with a warning**, and the analysis proceeds with the rest.
* If **none** of the list's species can be matched, the analysis **errors** instead of
  silently analysing every species.
* Previously, an unknown entry in a ``--slist`` file aborted the run with a library
  error; most such lists now succeed after reconciliation.

**What to do.** Legacy lists generally work unchanged. To require that every entry
match — turning any unmatched species into a hard error — use the new ``--strict``
flag:

.. code-block:: bash

   birdnet-analyze INPUT --slist my_list.txt --strict

In the GUI, selecting a species-list file shows a warning if it contains species the
chosen model does not have (only once that model has been downloaded).


Label languages differ between 2.4 and 3.0
------------------------------------------

**What changed.** The set of label languages (``--locale``) is not identical across
model versions. Nine languages available in 2.4 are **not** in 3.0:

   ``af``, ``ar``, ``en_uk``, ``hu``, ``it``, ``ko``, ``ro``, ``sl``, ``th``

Eleven languages are **new** in 3.0:

   ``bg``, ``ca``, ``cy``, ``es_ec``, ``es_es``, ``es_mx``, ``fa``, ``hr``, ``lt``,
   ``pt_pt``, ``sr``

**Consequence.** Requesting a locale the selected model version does not support falls
back to ``en_us``, with a warning. For example, ``--locale it`` with the default 3.0
model produces English labels.

**What to do.** Use a locale the model version supports, or switch to the 2.4 model if
you need one of its dropped languages. In the GUI the locale dropdown offers only the
selected model's languages.


Sensitivity has no effect on BirdNET 3.0
----------------------------------------

**What changed.** The 3.0 model applies its sigmoid inside the model graph and
outputs probabilities directly, so the sigmoid cannot be rescaled. ``--sensitivity``
therefore only applies to BirdNET 2.4 and custom classifiers (which run on the 2.4
base); Perch never used it.

**Consequence.** With the default 3.0 model a ``--sensitivity`` other than ``1.0`` is
ignored, with a warning. In the GUI the sensitivity slider is disabled while 3.0 or
Perch is selected.

**What to do.** Use ``--min_conf`` to tune 3.0 detections. If you rely on
sensitivity, select the 2.4 model (``--birdnet 2.4``).


Custom classifiers and training stay on the 2.4 base
----------------------------------------------------

**What changed / consequence.** Training and custom classifiers still use the 2.4 model
and its embeddings; 3.0 training is not yet supported. A custom classifier is therefore
loaded on the 2.4 base regardless of the default acoustic model. This behaviour is
unchanged, but is worth noting now that 3.0 is the default elsewhere.

The default output location for a trained classifier is now
``checkpoints/custom/Custom_Classifier`` **relative to the current working directory**
(previously an absolute path inside the installed package). Scripts that relied on the
old install-directory location will now write to, or look in, a different place; pass an
explicit ``-o/--output`` to pin a location. The new default is writable on a
pip-installed package and survives upgrades.


Resuming an interrupted analysis re-checks changed files
--------------------------------------------------------

**What changed.** A directory analysis now keeps a crash-safe journal so an interrupted
run can be continued: re-running the same analysis skips files that already completed.
Each file's stored result is keyed on its path **and its size and modification time**.

**Consequence.** When a run is resumed, an input file that changed on disk (edited,
regenerated, or merely touched so its modification time moved) since it was first
analysed is **re-analysed** rather than reusing the earlier result. Unchanged files are
skipped as before.


Other changes
-------------

* **Python 3.11+** is required (tested on 3.11–3.13).
* **Perch** is not available on macOS — it needs a TensorFlow build that is not yet
  published for that platform.
