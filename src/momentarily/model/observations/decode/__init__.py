"""Decoders: raw payload bytes to observation records.

Pure functions of the payload bytes and the fetch they came from
(``RawFetch``): no I/O, no clock. ``DECODER_VERSION`` names their output.

* ``decode_gtfsrt``: a GTFS-RT payload to its header, vehicles, trip updates
  and, for ``numbered``, its alert presence.
* ``decode_alerts_json``: the camsys alerts JSON to an alert presence.
* ``decode_ene``: an ENE outage payload to equipment observations.
"""

from momentarily.model.observations.decode.alerts import (
    decode_alerts_json,
    numbered_alert_entity,
)
from momentarily.model.observations.decode.common import (
    DECODER_VERSION,
    DecodeError,
    RawFetch,
)
from momentarily.model.observations.decode.ene import decode_ene, new_york_instant
from momentarily.model.observations.decode.gtfsrt import (
    GtfsRtRecords,
    decode_gtfsrt,
    parse_feed,
)

__all__ = [
    "DECODER_VERSION",
    "DecodeError",
    "GtfsRtRecords",
    "RawFetch",
    "decode_alerts_json",
    "decode_ene",
    "decode_gtfsrt",
    "new_york_instant",
    "numbered_alert_entity",
    "parse_feed",
]
