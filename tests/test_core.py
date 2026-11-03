from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.models.tvt_control_chain import ChainInputs
from src.models.tvt_horizon_response import summarize_horizon_response
from src.models.tvt_matched_chain import MatchedChainReference
from src.models.tvt_exponential_tail import ExponentialTail
from aeris.protocol import common_window_support, read_split, SPLITS


def test_circular_response_uses_shortest_signed_change():
    x = torch.zeros(2, 100, 22)
    x[:, :50, 20] = 179
    x[:, 50:, 20] = -179
    inputs = ChainInputs("alfa", dict(mean=[0.] * 22, std=[1.] * 22), .4)
    result = summarize_horizon_response(x, inputs, [])
    delta = x[:, 20:, 20] - x[:, :-20, 20]
    delta = torch.remainder(delta + 180, 360) - 180
    expected = np.stack([delta[:, i:i + 20].double().numpy().mean(1)
                         for i in range(61)], 1)
    np.testing.assert_allclose(result["ay_M"][:, 7], expected.mean(1), atol=1e-12)
    np.testing.assert_allclose(result["ay_V"][:, 7], expected.var(1), atol=1e-12)
    assert np.max(np.abs(expected)) <= 2


def test_reference_excludes_the_entire_query_source():
    n = 192
    library = {k: np.zeros((n, 2)) for k in ("history", "ca_context", "ay_context")}
    library["groups"] = np.repeat(["own", "other1", "other2"], 64)
    for edge in ("ca", "ay"):
        library[edge + "_M"] = np.r_[np.full((64, 1), 1000.), np.zeros((128, 1))]
        library[edge + "_V"] = np.ones((n, 1))
    reference = MatchedChainReference().fit(library)
    query = {k: v[:1].copy() for k, v in library.items()}
    query["ca_M"][:] = 0
    query["ay_M"][:] = 0
    evidence = reference.evidence(query)
    np.testing.assert_array_equal(evidence["ca_L"], [1.])
    np.testing.assert_array_equal(evidence["ay_L"], [1.])


def test_fixed_tail_has_no_empirical_ceiling_and_preserves_ties():
    reference = np.repeat(np.arange(100, dtype=float), 10)
    tail = ExponentialTail().fit(reference)
    at = np.array([40.])
    expected = -np.log((1 + (reference >= 40).sum()) / (len(reference) + 1) + 1e-12)
    np.testing.assert_array_equal(tail.transform(at), [expected])
    high = tail.transform(np.array([100., 1000.]))
    assert high[1] > high[0] > tail.anchor


def test_tail_failure_does_not_change_percentile():
    with pytest.raises(ValueError, match="20 strictly positive"):
        ExponentialTail().fit(np.zeros(1000))


def test_common_support_is_separate_from_model_window():
    refs = [SimpleNamespace(flight_index=0, start=i, end=i + 100) for i in range(105)]
    refs += [SimpleNamespace(flight_index=0, start=i, end=i + 100) for i in range(110, 113)]
    class Dataset:
        def __len__(self):
            return len(self.refs)
    dataset = Dataset()
    dataset.refs = refs
    valid = common_window_support(dataset)
    assert not valid[:99].any()
    assert valid[99:105].all()
    assert not valid[105:].any()
    assert refs[99].end == 199


def test_all_ten_fixed_splits_are_source_disjoint():
    for dataset, settings in SPLITS.items():
        for setting in settings:
            split, _ = read_split(dataset, setting)
            for role in ("train", "val", "test"):
                assert {split["flight_to_group"][f] for f in split[role]} == set(split["source_groups"][role])
