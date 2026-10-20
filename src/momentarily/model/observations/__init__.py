"""Layer 2: raw observations (schema ``observations/s1``).

What each upstream source said, when we asked, and what came back, failures
included. Upstream identities stay verbatim; matching to the network and every
judgment belong to layer 3.

``contracts`` holds the record types; ``keys`` the R2 key layout;
``columns`` the parquet column layout of the normalized records.
"""
