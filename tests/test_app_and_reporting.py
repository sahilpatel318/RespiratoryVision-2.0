import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from respiratory_vision import RespiratoryEstimator
from respiratory_vision.reporting import analysis_html, write_analysis_report

ROOT = Path(__file__).resolve().parents[1]


def test_report_and_missing_reference(tmp_path):
    result = RespiratoryEstimator().analyze(
        ROOT / "data/sample/synthetic_breathing_18bpm.mp4"
    )
    html = analysis_html(result)
    assert "accuracy cannot be calculated" in html
    assert "data:image/png;base64," in html
    assert "data:application/json;base64," in html
    write_analysis_report(result, tmp_path / "report.html", 18, "Synthetic fixture")
    assert (
        json.loads((tmp_path / "report.json").read_text())["report_reference_bpm"] == 18
    )
    assert (tmp_path / "report.csv").read_text().startswith("seconds,")


def test_streamlit_initialize_and_analyze():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    assert len(app.tabs) == 4
    next(b for b in app.button if b.label == "Run analysis").click().run(timeout=90)
    assert not app.exception
    assert app.session_state["analysis"]["quality"]["state"] == "ACCEPTED"
    assert abs(app.session_state["analysis"]["estimate"]["bpm"] - 18) < 2


def test_streamlit_demo_benchmark():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    next(b for b in app.button if b.label == "Run demo benchmark").click().run(
        timeout=120
    )
    assert not app.exception
    assert app.session_state["benchmark"]["provenance"]["case_count"] == 4
    assert app.session_state["benchmark"]["methods"]["quality_fusion"]["n_total"] == 4
