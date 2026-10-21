"""Backfill adapters: the pre-recorder archives as legacy-lineage records.

One module per legacy source (doc ``model-observations.md`` §5): ``trace``,
``fresh_feeds`` (``archive/trip_updates`` and ``archive/vehicles``),
``alerts``, ``alerts_liveness``, ``predictions``, ``ene`` and ``health``.
Each keeps what its archive kept and marks every other field ``Unknown``;
nothing is inferred. ``run`` drives them over a local mirror and writes
parquet (``python -m momentarily.model.observations.legacy``, needs the
``derive`` extra). The adapters themselves need no ``pyarrow``.
"""
