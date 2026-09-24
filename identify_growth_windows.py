"""Find completed, reviewable short growth windows in archived daily closes.

These are retrospective labels for chart review, not trading signals.  All
fundamental and liquidity checks are evaluated at the window's first close.
"""

from __future__ import annotations

from datetime import date
from typing import Callable


MIN_SESSIONS = 4
MAX_SESSIONS = 9
TARGET_RETURN = 0.20
MAX_ONE_DAY_GAIN = 0.12
FOLLOW_THROUGH_SESSIONS = 2
MIN_RETAINED_GAIN = 0.10


def find_growth_windows(
    bars: list[dict | None],
    eligible_at: Callable[[int], bool],
    first_index: int = 0,
) -> list[dict]:
    """Return nonoverlapping windows, indexed into *bars*.

    A window ends at the first close 20% above its starting close.  An earlier
    crossing is rejected even if the price remains high four days later.  A
    valid window also has no daily close jump above 12%, and retains at least
    half the gain at both of the following closes.  The latter rule means the
    latest two sessions cannot yet be used as completed endpoints.
    """
    candidates = []
    last_start = len(bars) - MIN_SESSIONS - FOLLOW_THROUGH_SESSIONS - 1
    for start in range(max(0, first_index), max(0, last_start + 1)):
        start_bar = bars[start]
        if not start_bar or float(start_bar["c"]) <= 0:
            continue
        start_close = float(start_bar["c"])
        first_crossing = None
        excessive_jump = False
        for end in range(start + 1, min(start + MAX_SESSIONS + 1,
                                        len(bars) - FOLLOW_THROUGH_SESSIONS)):
            previous, current = bars[end - 1], bars[end]
            if not previous or not current or float(previous["c"]) <= 0:
                break
            if float(current["c"]) / float(previous["c"]) - 1 > MAX_ONE_DAY_GAIN:
                excessive_jump = True
            if float(current["c"]) / start_close - 1 >= TARGET_RETURN:
                first_crossing = end
                break
        if first_crossing is None or not MIN_SESSIONS <= first_crossing - start <= MAX_SESSIONS:
            continue
        calendar_days = (date.fromisoformat(bars[first_crossing]["d"]) -
                         date.fromisoformat(start_bar["d"])).days
        if not 3 < calendar_days < 14:
            continue
        if excessive_jump:
            continue
        follow = bars[first_crossing + 1:first_crossing + 1 + FOLLOW_THROUGH_SESSIONS]
        if len(follow) != FOLLOW_THROUGH_SESSIONS or any(
            bar is None or float(bar["c"]) / start_close - 1 < MIN_RETAINED_GAIN
            for bar in follow
        ):
            continue
        if not eligible_at(start):
            continue
        end_close = float(bars[first_crossing]["c"])
        candidates.append({
            "start_index": start,
            "end_index": first_crossing,
            "sessions": first_crossing - start,
            "calendar_days": calendar_days,
            "start_date": start_bar["d"],
            "end_date": bars[first_crossing]["d"],
            "start_close": start_close,
            "end_close": end_close,
            "return": end_close / start_close - 1,
        })

    # Neighboring start dates describe the same rise.  Keep the strongest
    # representative and allow further, nonoverlapping rises for that stock.
    chosen = []
    for candidate in sorted(candidates, key=lambda x: (-x["return"], x["start_index"])):
        if all(candidate["end_index"] < old["start_index"] or
               candidate["start_index"] > old["end_index"] for old in chosen):
            chosen.append(candidate)
    return sorted(chosen, key=lambda x: x["start_index"])
