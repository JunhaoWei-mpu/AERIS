"""Reference-compatible VUS-ROC and VUS-PR.

This compact NumPy implementation follows ``RangeAUC_volume_opt`` from the
official ``vus==0.0.5`` package (MIT license), while adding input validation
and explicit handling of degenerate labels.  A regression fixture in
``tests/test_taes_protocol.py`` is checked against the official wheel.

Reference:
    Paparrizos et al., "Volume Under the Surface: A New Accuracy Evaluation
    Measure for Time-Series Anomaly Detection," PVLDB 15(11), 2022.
"""
from __future__ import annotations

import numpy as np


def _ranges(label: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(label) > 0
    delta = np.diff(np.concatenate(([False], binary, [False])).astype(np.int8))
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _expanded_sequences(
    length: int,
    sequences: list[tuple[int, int]],
    window: int,
) -> list[tuple[int, int]]:
    """Match the official non-overlapping expanded-sequence construction."""
    if not sequences:
        return []
    left = max(sequences[0][0] - window // 2, 0)
    output: list[tuple[int, int]] = []
    for current, following in zip(sequences[:-1], sequences[1:]):
        current_right = current[1] + window // 2
        following_left = following[0] - window // 2
        if current_right < following_left:
            output.append((left, current_right))
            left = following_left
    output.append(
        (
            left,
            min(sequences[-1][1] + window // 2, length - 1),
        )
    )
    return output


def _soft_labels(
    original: np.ndarray,
    sequences: list[tuple[int, int]],
    window: int,
) -> np.ndarray:
    labels = original.astype(np.float64, copy=True)
    if window <= 0:
        return labels
    length = labels.size
    for start, end in sequences:
        right = np.arange(end + 1, min(end + window // 2 + 1, length))
        labels[right] += np.sqrt(1.0 - (right - end) / window)
        left = np.arange(max(start - window // 2, 0), start)
        labels[left] += np.sqrt(1.0 - (start - left) / window)
    return np.minimum(1.0, labels)


def vus_roc_pr_reference(
    y_true: np.ndarray,
    score: np.ndarray,
    max_buffer: int = 100,
    thresholds: int = 250,
) -> tuple[float, float]:
    """Return official-definition VUS-ROC and VUS-PR.

    ``max_buffer`` is the largest evaluated range width.  Every integer width
    from zero through ``max_buffer`` enters the volume average.
    """
    return vus_roc_pr_multiseries(
        [y_true],
        [score],
        max_buffer=max_buffer,
        thresholds=thresholds,
    )


def vus_roc_pr_multiseries(
    y_true_series: list[np.ndarray],
    score_series: list[np.ndarray],
    max_buffer: int = 100,
    thresholds: int = 250,
) -> tuple[float, float]:
    """Generalize the reference surface without crossing flight boundaries."""
    if len(y_true_series) != len(score_series) or not y_true_series:
        raise ValueError("labels and scores must contain the same nonzero series")
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for y_true, score in zip(y_true_series, score_series):
        labels = np.asarray(y_true, dtype=np.float64)
        scores = np.asarray(score, dtype=np.float64)
        if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
            raise ValueError("each label/score series must be aligned and 1-D")
        if labels.size == 0 or not np.isfinite(scores).all():
            raise ValueError("each VUS series requires finite scores")
        prepared.append(((labels != 0).astype(np.float64), scores))
    if sum(labels.size for labels, _ in prepared) < 2:
        raise ValueError("VUS requires at least two scores")
    if max_buffer < 0 or thresholds < 2:
        raise ValueError("invalid VUS surface resolution")
    positives = float(sum(labels.sum() for labels, _ in prepared))
    total_length = int(sum(labels.size for labels, _ in prepared))
    if positives == 0.0 or positives == total_length:
        return float("nan"), float("nan")
    structures = [
        (
            labels,
            _ranges(labels),
        )
        for labels, _ in prepared
    ]
    fixed_ranges = [
        _expanded_sequences(labels.size, sequences, max_buffer)
        for labels, sequences in structures
    ]

    pooled_scores = np.concatenate([score for _, score in prepared])
    sorted_scores = -np.sort(-pooled_scores)
    threshold_indices = np.linspace(
        0, pooled_scores.size - 1, thresholds
    ).astype(int)
    threshold_values = sorted_scores[threshold_indices]
    predictions = [
        np.stack([score >= value for value in threshold_values], axis=0)
        for _, score in prepared
    ]
    n_predicted = sum(
        prediction.sum(axis=1).astype(np.float64)
        for prediction in predictions
    )
    auc_values = np.zeros(max_buffer + 1, dtype=np.float64)
    ap_values = np.zeros(max_buffer + 1, dtype=np.float64)

    for window in range(max_buffer + 1):
        soft = [
            _soft_labels(labels, sequences, window)
            for labels, sequences in structures
        ]
        active_ranges = [
            _expanded_sequences(labels.size, sequences, window)
            for labels, sequences in structures
        ]
        n_labels = sum(
            float(labels[start : end + 1].sum())
            for labels, ranges in zip(soft, fixed_ranges)
            for start, end in ranges
        )
        tpr_fpr = np.zeros((thresholds + 2, 2), dtype=np.float64)
        precision = np.ones(thresholds + 1, dtype=np.float64)
        p_effective = (positives + n_labels) / 2.0
        n_effective = total_length - p_effective
        n_active_ranges = sum(len(ranges) for ranges in active_ranges)

        for index in range(thresholds):
            true_positive = sum(
                float(
                    np.dot(
                        labels[start : end + 1],
                        prediction[index, start : end + 1],
                    )
                )
                for labels, prediction, ranges in zip(
                    soft, predictions, fixed_ranges
                )
                for start, end in ranges
            )
            false_positive = n_predicted[index] - true_positive
            existence = sum(
                np.dot(
                    labels[start : end + 1],
                    prediction[index, start : end + 1],
                )
                > 0.0
                for labels, prediction, ranges in zip(
                    soft, predictions, active_ranges
                )
                for start, end in ranges
            )
            existence_ratio = existence / n_active_ranges
            recall = min(true_positive / p_effective, 1.0)
            tpr_fpr[index + 1] = (
                recall * existence_ratio,
                false_positive / n_effective,
            )
            precision[index + 1] = true_positive / n_predicted[index]

        tpr_fpr[-1] = (1.0, 1.0)
        roc_width = np.diff(tpr_fpr[:, 1])
        roc_height = (tpr_fpr[1:, 0] + tpr_fpr[:-1, 0]) / 2.0
        auc_values[window] = float(np.dot(roc_width, roc_height))
        pr_width = tpr_fpr[1:-1, 0] - tpr_fpr[:-2, 0]
        pr_height = (precision[1:] + precision[:-1]) / 2.0
        ap_values[window] = float(np.dot(pr_width, pr_height))

    return float(auc_values.mean()), float(ap_values.mean())
