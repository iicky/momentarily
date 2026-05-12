"""Offline HMM training + calibration toolkit.

Reads the collector's accumulated JSONL archive, runs the forward filter,
and (eventually) trains transition matrices via Baum-Welch EM. Pairs with the
TS Cloudflare Worker (live publisher) via the schema contract — this side is
where the math happens; the Worker just consumes parameters.
"""
