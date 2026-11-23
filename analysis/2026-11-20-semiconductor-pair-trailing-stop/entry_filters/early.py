"""Three fixed, nested pre-entry filters using only completed five-minute bars."""

from datetime import timedelta
from math import fsum


SYMBOLS = ("SOXL", "SOXS")
RULES = ("early_trend", "early_reclaim", "early_breakout")
STEP = timedelta(minutes=5)


def early_masks(tape, times):
    """At grid index i, return decisions derived from bars strictly before i.

    The caller supplies aligned, validated OHLC tuples and applies the original
    five-bar entry rule in addition to these masks.
    """
    count = len(times)
    if any(len(tape[symbol]) != count for symbol in SYMBOLS):
        raise ValueError("Both ETF series must match the decision grid")
    masks = {rule: {s: [False] * count for s in SYMBOLS} for rule in RULES}
    closes = {s: [bar[3] for bar in tape[s]] for s in SYMBOLS}
    segment_start = 0
    for i, at in enumerate(times):
        if i and at - times[i - 1] != STEP:
            segment_start = i
        if i - segment_start < 12:
            continue
        for symbol, opposite in (("SOXL", "SOXS"), ("SOXS", "SOXL")):
            close = closes[symbol]
            last = close[i - 1]
            earlier = fsum(close[i - 12:i - 6])
            recent = fsum(close[i - 6:i])
            if not (recent > earlier and last > recent / 6):
                continue
            masks["early_trend"][symbol][i] = True
            had_pullback = any(close[j] < close[j - 1] for j in range(i - 4, i - 1))
            if not (had_pullback and last > tape[symbol][i - 2][1]):
                continue
            masks["early_reclaim"][symbol][i] = True
            prior_high = max(bar[1] for bar in tape[symbol][i - 4:i - 1])
            inverse = closes[opposite]
            masks["early_breakout"][symbol][i] = last > prior_high and inverse[i - 1] < inverse[i - 4]
    return masks
