GPU inference
===============================

Analysis can run on an NVIDIA GPU. Select the device with ``--device`` on the command
line, with the *Device* setting under the computing settings of the *Multi-file
analysis* tab, or with ``analyze(..., device="GPU")`` in the Python API. Accepted
values are ``CPU``, ``GPU`` and ``GPU:<index>`` for a specific card.

The device is checked before an analysis starts. If the GPU cannot be used, the run
continues on the CPU and reports why in the log instead of failing.

Which models support it
--------------------------------------------------------

Only **BirdNET 3.0**. It is the one model this package runs on ONNX Runtime, which is
the backend that can dispatch to a GPU. BirdNET 2.4, custom classifiers (which run on
the 2.4 base) and Perch are always analyzed on the CPU, and requesting a GPU for them
logs a warning and falls back.

Installing a GPU-capable ONNX Runtime
--------------------------------------------------------

The ``onnxruntime`` wheel installed with BirdNET-Analyzer is a CPU-only build. GPU
inference needs the CUDA build in its place:

.. code-block:: bash

   pip uninstall onnxruntime
   pip install onnxruntime-gpu

``onnxruntime-gpu`` does not bundle CUDA, and **which CUDA it needs depends on the
release**: 1.29 is built against CUDA 13, releases up to 1.24 against CUDA 12. The
`ONNX Runtime CUDA requirements <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_
page lists the pairing per release. Without a system-wide CUDA installation, the
``nvidia-*`` pip packages provide the libraries.

That pairing also decides which cards can be used: CUDA 13 dropped Maxwell, Pascal and
Volta, so a card older than Turing (compute capability below 7.5) needs a CUDA 12
release of ONNX Runtime and a cuDNN built for it:

.. code-block:: bash

   pip install "onnxruntime-gpu==1.24.4" "nvidia-cudnn-cu12==9.8.0.87" nvidia-cublas-cu12

Mismatches show up in two ways. A CUDA runtime that does not match the ONNX Runtime
release fails to load the provider
(``Error loading onnxruntime_providers_cuda.dll which depends on cublasLt64_NN.dll``)
and leaves the session on the CPU provider. A cuDNN too new for the card creates the
session but fails during inference with
``CUDNN failure 5003: CUDNN_STATUS_EXECUTION_FAILED_CUDART``.

On Windows the ``nvidia-*`` packages put their libraries in ``site-packages/nvidia``
rather than on PATH, where ONNX Runtime looks for them. BirdNET-Analyzer adds those
directories to PATH itself when a GPU is requested, so no manual setup is needed.

.. warning::
   DirectML (``onnxruntime-directml``) is not a working alternative: the BirdNET 3.0
   acoustic model fails on it at every batch size and precision. Only the CUDA
   execution provider is accepted as a GPU, so a DirectML-only installation reports no
   GPU and runs on the CPU rather than failing mid-analysis.

Checking that the GPU is being used
--------------------------------------------------------

``--device GPU`` being accepted only means the installed ONNX Runtime is a CUDA build.
If its libraries do not load, ONNX Runtime falls back to the CPU provider on its own.
``nvidia-smi`` during a run is the reliable check: a GPU analysis shows the worker
process holding video memory. Debug logging (``-v``) additionally reports the provider
the model was loaded with.

Batch size and workers
--------------------------------------------------------

**Raise the batch size for GPU runs.** At the default of 1 a GPU run is no faster than
the CPU. The GUI switches the batch size to 16 together with the device; on the
command line, set ``--batch_size`` explicitly. An analysis started on a GPU with a
batch size of 1 logs a reminder.

The worker default is **1** on a GPU, against one per core on the CPU, because each
worker is a separate process holding its own copy of the model in video memory.
Raising ``--n_workers`` can still improve throughput on a card with enough memory.
