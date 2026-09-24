"""Date-grouped chronological folds with label-window and availability purging."""

from dataclasses import dataclass
from datetime import datetime

from .contracts import Sample, require_aware
from .timeaxis import ET


@dataclass(frozen=True)
class Fold:
    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    excluded: dict[str, str]


def purged_fold(samples: list[Sample], validation_start: datetime,
                validation_end: datetime, evaluation_as_of: datetime) -> Fold:
    require_aware(validation_start, validation_end, evaluation_as_of)
    for boundary in (validation_start, validation_end):
        local = boundary.astimezone(ET)
        if (local.hour, local.minute, local.second, local.microsecond) != (0, 0, 0, 0):
            raise ValueError("fold boundaries must be ET session-date midnights")
    if validation_start >= validation_end or evaluation_as_of < validation_end:
        raise ValueError("invalid validation/evaluation range")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("duplicate sample id")
    train, validation, excluded = [], [], {}
    for sample in sorted(samples, key=lambda s: (s.decision_at, s.symbol, s.sample_id)):
        if sample.label_status != "mature":
            excluded[sample.sample_id] = "unmatured_or_incomplete"
        elif sample.decision_at < validation_start:
            if sample.label_end_at >= validation_start or sample.label_available_at >= validation_start:
                excluded[sample.sample_id] = "purged_label_overlap_or_late_availability"
            else:
                train.append(sample)
        elif sample.decision_at < validation_end:
            if sample.label_available_at <= evaluation_as_of:
                validation.append(sample)
            else:
                excluded[sample.sample_id] = "outcome_unavailable_at_evaluation"
        else:
            excluded[sample.sample_id] = "outside_fold"
    return Fold(tuple(train), tuple(validation), excluded)
