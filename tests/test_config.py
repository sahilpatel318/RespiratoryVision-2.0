import pytest

from respiratory_vision.config import PipelineConfig


def test_default_config_valid():
    PipelineConfig().validate()


def test_invalid_roi_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(roi_x=0.8, roi_w=0.5).validate()
