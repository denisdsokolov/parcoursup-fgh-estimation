import numpy as np

from preference_features import log_count


def test_log_count_handles_scalar_zero_and_positive_counts():
    assert log_count(0) == 0.0
    assert log_count(1) == 0.0
    np.testing.assert_allclose(log_count(np.e), 1.0, rtol=1e-14, atol=0.0)


def test_log_count_vectorizes_and_clamps_zero():
    transformed = log_count([0, 1, 10])

    np.testing.assert_allclose(transformed, [0.0, 0.0, np.log(10)])
