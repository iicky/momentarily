# Feed-expansion overhead matrix

A reference card for the engineering cost of adding each additional transit
source to Momentarily. Use it to scope future expansion work.

Effort is in **engineer-days** (one focused day each, not calendar days), against
the current TypeScript Worker + R2 publisher (see [ADR 0001](adr/0001-cloudflare-workers-r2-only-split-ts-python.md)).
Subway alerts and elevator/escalator state ship today.

| Source | Effort | What you need | What you get | Gotchas |
|---|---|---|---|---|
| Subway alerts | 0 (shipped) | MTA key | Per-line + per-stop alerts | Done |
| E&E (elevators/escalators) | 0 (shipped) | MTA key | Outage list, ADA-pathway flag | Done |
| LIRR alerts | ~1 | Same MTA key | LIRR line status, station alerts | Different alert vocab ("On or Close" vs "Good Service") — extend the mapping |
| MNR alerts | ~1 | Same MTA key | MNR line status, station alerts | Same as LIRR; trivial once LIRR is done |
| Bus alerts | ~1–2 | Same MTA key | Per-bus-route status | Nov 2025 backend auth issues; verify the gateway is stable before committing |
| PATH | ~3–5 | **PANYNJ** dev account (not MTA) | PATH line status, station alerts | Different upstream provider + TOS; GTFS-RT at panynj.gov; new key to manage |
| NYC Ferry | ~3–5 | NYC EDC GTFS (free, no auth) | Ferry route status, alerts | Hornblower-operated for NYC EDC; reliability spotty. **SI Ferry is separate** (NYC DOT) — adds ~2 more days |
| Subway/bus/RR trip-updates | ~5–8 | Same MTA key | Real-time ETAs at stations | Protobuf parsing; NYCT/MTA-RR proto extensions add fields. Partly underway — see the trip-updates archive work |
| Bridge/Tunnel travel times | ~7–14 + $$ | Google Directions / Waze CCP / INRIX | Live travel time per crossing | **No MTA feed.** Paid API (~$5/1k req ≈ $50/mo at a 5-min cron × 6 crossings); TOS review for redistribution. Waze CCP is free but contribution-required |

## Reading the matrix

- **Cheap wins (1–3 days each):** LIRR, MNR alerts. Same gateway, same key — roughly a handful of fetch URLs plus a mapping extension.
- **Modest add (1–2 days):** Bus alerts. Hold until the Nov 2025 auth issues are confirmed resolved.
- **Real projects (~1 week each):** PATH or NYC Ferry. New agency + TOS + auth surface.
- **Major project (~1–2 weeks + ongoing $):** B&T travel times. The data isn't free; pick a paid provider. The schema scaffolding (`bridges`, `tunnels`) already exists in the snapshot contract.

## Notes

- The alert-shaped sources (LIRR, MNR, bus, PATH, ferry) all reduce to the same
  pattern already in place for subway: fetch a GTFS-RT alerts feed, map its
  `alert_type` vocabulary to the coarse status buckets, derive per-route and
  per-stop views. The cost is mostly the new vocabulary and the new auth surface.
- Trip-updates is the one source that unlocks a genuinely new capability
  (station ETAs and an independent recovery signal) rather than more of the same
  alert surface — which is why it carries protobuf complexity the alert feeds don't.
- Geographic rendering (station coordinates, route shapes, crossing centroids) is
  tracked separately from feed expansion.
