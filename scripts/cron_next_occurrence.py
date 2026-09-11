from __future__ import annotations

import datetime
import re
from datetime import timedelta

_DOW: dict[str, int] = {
    "SUN": 6,
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
}


def next_cron_occurrence(cron_expr: str, after: datetime.datetime) -> datetime.datetime:
    """Return next UTC occurrence of *cron_expr* strictly after *after*.

    Supported: integer minute/hour, '*' dom and month, integer 0-7 or 3-letter
    abbrev DOW.  Raises ValueError for unsupported forms.
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5-field cron, got {len(parts)}: {cron_expr!r}")
    minute_s, hour_s, dom_s, month_s, dow_s = parts
    if not minute_s.isdigit():
        raise ValueError(f"Unsupported cron minute: {minute_s!r}")
    minute = int(minute_s)
    if not (0 <= minute <= 59):
        raise ValueError(f"Cron minute out of range 0-59: {minute}")
    if not hour_s.isdigit():
        raise ValueError(f"Unsupported cron hour: {hour_s!r}")
    hour = int(hour_s)
    if not (0 <= hour <= 23):
        raise ValueError(f"Cron hour out of range 0-23: {hour}")
    if dom_s != "*":
        raise ValueError(f"Unsupported cron day-of-month (only '*'): {dom_s!r}")
    if month_s != "*":
        raise ValueError(f"Unsupported cron month (only '*'): {month_s!r}")
    if dow_s.upper() in _DOW:
        target = _DOW[dow_s.upper()]
    elif dow_s.isdigit():
        d = int(dow_s)
        if not (0 <= d <= 7):
            raise ValueError(f"Cron DOW integer out of range 0-7: {d}")
        target = 6 if d in (0, 7) else d - 1
    else:
        raise ValueError(f"Unsupported cron DOW: {dow_s!r}")
    today = after.astimezone(datetime.UTC).date()
    days_ahead = (target - today.weekday()) % 7
    c = today + timedelta(days=days_ahead)
    candidate = datetime.datetime(
        c.year, c.month, c.day, hour, minute, tzinfo=datetime.UTC
    )
    if candidate <= after:
        candidate += timedelta(weeks=1)
    return candidate


def parse_crons(toml_text: str) -> list[str]:
    """Extract quoted cron strings from the crons = [...] line in wrangler.toml."""
    m = re.search(r"^\s*crons\s*=\s*(\[[^\]]*\])", toml_text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not locate 'crons = [...]' in wrangler.toml")
    exprs = re.findall(r'"([^"]+)"', m.group(1))
    if not exprs:
        raise RuntimeError("crons array contains no quoted expressions")
    return exprs
