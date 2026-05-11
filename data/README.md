# Local data corpus

The collector container writes here. Contents are git-ignored — never committed.

Layout once the collector is running:

```
data/
├── alerts/YYYY-MM-DD.jsonl    # one line per alert observation per poll
├── ene/YYYY-MM-DD.jsonl       # one line per equipment status observation per hourly poll
└── meta/
    ├── last_fetched.json      # latest successful fetch per feed
    └── poll_log.jsonl         # one line per poll attempt (success or error)
```

Storage estimate: ~125 KB/day raw, ~45 MB/year. Trivial.

To start collecting, see [`../collector/README.md`](../collector/README.md).
