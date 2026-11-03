"""Layer 3: events (``events/s1``).

Built so far: the per-service-day match index the trip matcher reads
(``match_index``), the storage keys of the index and of the event files
(``keys``), the route-to-feed table (``route_feed``), the public event
contracts (``contracts``) with their pinned parquet layouts (``columns``),
and ``python -m momentarily.model.events``, which builds the index from a
local layer-1 root. Nothing builds event records yet.
"""
