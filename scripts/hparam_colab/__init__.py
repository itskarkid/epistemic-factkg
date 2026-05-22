"""hparam_colab — environment-aware hyperparameter search launcher.

Detects whether we are running inside Google Colab, on a local GPU machine,
or on a CPU-only machine, and acts accordingly:

  Colab / local GPU  → runs src.pipeline.model.hparam_search directly
  CPU-only local     → builds colab_upload.zip + generates notebook + opens it
"""
