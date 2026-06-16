# Momentarily publisher (Cloudflare Worker)

The live publish path: a TypeScript Worker that fetches MTA GTFS-RT feeds on a
Cron Trigger, runs the HMM derivation, and writes the snapshot to R2.
[`src/momentarily/`](../src/momentarily/) is the offline reference implementation
of the same derivation logic.

## alert_type → status mapping

MTA's alerts feed uses an open-set string for `alert_type` (`Delays`,
`Service Change`, `Slow Speeds`, `Trains Rerouted`, `Planned – Multiple Changes`,
…). New values can appear without versioning, so the mapping is maintained by
hand and unknown values pass through as their own raw label rather than being
dropped or coerced.

Two axes come out of the mapping, both in [`src/mapping.ts`](src/mapping.ts):

- **`coarseStatus(alertType)`** — a short human label, chosen to preserve the
  entity vocabulary the `homeassistant-mta-subway` integration shipped before the
  Momentarily migration. First substring match wins, so the table is ordered
  most-specific first.
- **`categoryForLabel(label)`** — a stable token for the coarse label, the
  `category` axis on the snapshot. Derived from the label so there is one table to
  maintain, not two.

### Coarse status table

The live table matches by substring (first match wins):

| Substring                | Status         |
| ------------------------ | -------------- |
| `Planned -`              | `Planned Work` |
| `Suspend`                | `Suspended`    |
| `No Trains`              | `Suspended`    |
| `No Scheduled Service`   | `Suspended`    |
| `Severe Delays`          | `Delays`       |
| `Delays`                 | `Delays`       |
| `Reroute`                | `Service Change` |
| `Trains Rerouted`        | `Service Change` |
| `Stops Skipped`          | `Service Change` |
| `Express to Local`       | `Service Change` |
| `Local to Express`       | `Service Change` |
| `Service Change`         | `Service Change` |
| `Boarding Change`        | `Service Change` |
| `Slow Speeds`            | `Slow Speeds`  |
| `Station Notice`         | `Information`   |
| `Special Schedule`       | `Information`   |
| `Information`            | `Information`   |
| _no match_               | raw `alert_type` (passed through) |
| _null / empty_           | `Good Service` (`NO_ALERTS_FALLBACK`) |

### Label → category

| Coarse label   | Category             |
| -------------- | -------------------- |
| `Good Service` | `none`               |
| `Planned Work` | `planned_work`       |
| `Delays`       | `delays`             |
| `Service Change` | `service_change`   |
| `Suspended`    | `service_suspension` |
| `Slow Speeds`  | `slow_speeds`        |
| `Information`  | `information`        |
| _anything else_ | `other`             |

## Relationship to mapping.py

[`src/momentarily/mapping.py`](../src/momentarily/mapping.py) is the offline
reference. It matches `alert_type` by exact dict lookup (plus `Planned*` and
`No … Service` prefix rules); the Worker matches by substring and is the live,
more-complete table. They agree on every `alert_type` currently observed in
production. Known intentional divergences:

- `Slow Speeds` → `Delays` in Python (locked by `tests/test_mapping.py`) vs its
  own `Slow Speeds` label in the Worker, which has a dedicated `slow_speeds`
  category.
- Python carries railroad-only types (`Cancellations`, `Track Change`,
  `Weather`) and `Some Delays`, which the subway-only Worker does not need.

## Maintenance

Revisit this table on each publisher release. The unmapped-`alert_type` rate is
tracked by the offline drift job
([`training/drift.py`](../training/drift.py), via `is_known_alert_type`) over the
predictions stream the Worker writes — a rising rate is the signal that MTA added
a value that needs a row here. When that happens, add the mapping in both
`mapping.ts` and `mapping.py`, extend the table above, and ship a release.
