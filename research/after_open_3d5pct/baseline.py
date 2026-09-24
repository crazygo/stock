"""B0: a train-only empirical rate. It is not a fitted predictive feature model."""

import math

from .contracts import Sample


def fit_base_rate(train: tuple[Sample, ...]) -> float:
    if not train or any(s.label_status != "mature" or s.target not in (0, 1) for s in train):
        raise ValueError("baseline requires nonempty, fully matured training outcomes")
    return sum(s.target for s in train) / len(train)


def score_constant(probability: float, validation: tuple[Sample, ...]) -> dict:
    if not validation or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("nonempty validation and finite probability required")
    if any(s.label_status != "mature" or s.target not in (0, 1) for s in validation):
        raise ValueError("evaluation requires matured outcomes")
    p = max(1e-12, min(1 - 1e-12, probability))
    n = len(validation)
    return {
        "n": n,
        "observed_rate": sum(s.target for s in validation) / n,
        "brier": sum((probability - s.target) ** 2 for s in validation) / n,
        "log_loss": -sum(s.target * math.log(p) + (1 - s.target) * math.log(1 - p) for s in validation) / n,
    }
