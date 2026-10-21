"""The offline derive job: recorded raw objects to derived parquet files.

``storage`` is the read/write interface (``LocalStorage`` here; the R2 store
is ``training.derive_observations``), ``bundle`` reads tick bundles and
blobs, ``alert_index`` builds the daily AlertVersion index from presences,
and ``run`` derives one UTC day (``python -m
momentarily.model.observations.derive``, needs the ``derive`` extra). Only
``run`` and ``cli`` import ``pyarrow``.
"""
