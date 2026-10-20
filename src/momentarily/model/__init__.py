"""Layered data-model contracts, built beside the live pipeline.

Layer 1 is the network (``momentarily.model.network``); every later layer keys
off its identities. ``momentarily.model.conventions`` holds the time, interval,
missing-data, and provenance rules all layers share.
"""
