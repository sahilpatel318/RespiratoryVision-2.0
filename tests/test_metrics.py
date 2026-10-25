from respiratory_vision.metrics import regression_metrics


def test_regression_metrics():
    m = regression_metrics([10, 20], [11, 18])
    assert abs(m.mae - 1.5) < 1e-9
    assert abs(m.mean_bias + 0.5) < 1e-9
