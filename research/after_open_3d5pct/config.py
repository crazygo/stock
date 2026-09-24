"""Frozen v1 contract. A change of scientific meaning requires a new version."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    schema_version: int
    strategy_id: str
    timezone: str
    decision_hours_et: tuple[str, ...]
    target_return: float
    horizon_regular_minutes: int
    entry_bar_minutes: int
    signal_latency_seconds: int
    manual_delay_seconds: int
    entry_max_wait_minutes: int
    lookback_sessions: tuple[int, ...]
    universe_mode: str
    seed: int

    def __post_init__(self):
        expected = {
            "schema_version": 1,
            "strategy_id": "after_open_3d5pct_v1",
            "timezone": "America/New_York",
            "decision_hours_et": ("11:30", "12:30", "13:30", "14:30"),
            "target_return": 0.05,
            "horizon_regular_minutes": 1170,
            "entry_bar_minutes": 5,
            "lookback_sessions": (1, 3, 5, 10, 20),
            "universe_mode": "point_in_time_required",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name}: v1 semantics are frozen; create a new contract version")
        for name in ("signal_latency_seconds", "manual_delay_seconds", "entry_max_wait_minutes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    raw = json.loads(path.read_text())
    for key in ("decision_hours_et", "lookback_sessions"):
        raw[key] = tuple(raw[key])
    return Config(**raw)
