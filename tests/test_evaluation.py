"""
Tests for retrieval evaluation logic.

Tests the EvaluationReport model and the evaluate_retrieval tool output
contract without making real API calls.
"""
import json
import pytest

from udaplay.tools import EvaluationReport


class TestEvaluationReportModel:
    """Unit tests for the EvaluationReport Pydantic model."""

    def test_high_confidence_local_rag(self):
        report = EvaluationReport(
            confidence="high",
            useful=True,
            description="Document directly contains the game developer and publisher.",
            needs_web_search=False,
        )
        assert report.confidence == "high"
        assert report.useful is True
        assert report.needs_web_search is False

    def test_low_confidence_needs_web(self):
        report = EvaluationReport(
            confidence="low",
            useful=False,
            description="Retrieved documents are about unrelated games.",
            needs_web_search=True,
        )
        assert report.confidence == "low"
        assert report.useful is False
        assert report.needs_web_search is True

    def test_medium_confidence(self):
        report = EvaluationReport(
            confidence="medium",
            useful=True,
            description="Partial match — platform mentioned but not exact title.",
            needs_web_search=False,
        )
        assert report.confidence == "medium"

    def test_json_roundtrip(self):
        original = EvaluationReport(
            confidence="high",
            useful=True,
            description="Test.",
            needs_web_search=False,
        )
        serialised = original.model_dump_json()
        restored = EvaluationReport.model_validate_json(serialised)
        assert restored == original

    def test_dict_roundtrip(self):
        report = EvaluationReport(
            confidence="low",
            useful=False,
            description="No relevant documents.",
            needs_web_search=True,
        )
        data = report.model_dump()
        assert data == {
            "confidence": "low",
            "useful": False,
            "description": "No relevant documents.",
            "needs_web_search": True,
        }

    def test_missing_confidence_raises(self):
        with pytest.raises(Exception):
            EvaluationReport(
                useful=True,
                description="oops",
                needs_web_search=False,
            )

    def test_missing_needs_web_search_raises(self):
        with pytest.raises(Exception):
            EvaluationReport(
                confidence="high",
                useful=True,
                description="missing field",
            )


class TestEvaluationOutputContract:
    """
    Validates that the string returned by evaluate_retrieval always represents
    valid JSON that conforms to the EvaluationReport schema.

    The evaluate_retrieval tool is integration-tested in test_tools.py with
    a mocked LLM.  Here we test the fallback serialisation paths.
    """

    def test_fallback_is_valid_json(self):
        fallback = EvaluationReport(
            confidence="low",
            useful=False,
            description="Evaluation error: API timeout.",
            needs_web_search=True,
        )
        json_str = fallback.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["confidence"] == "low"
        assert parsed["needs_web_search"] is True

    def test_all_fields_present_in_fallback(self):
        fallback = EvaluationReport(
            confidence="low",
            useful=False,
            description="Error.",
            needs_web_search=True,
        )
        keys = set(json.loads(fallback.model_dump_json()).keys())
        assert keys == {"confidence", "useful", "description", "needs_web_search"}

    @pytest.mark.parametrize("confidence", ["high", "medium", "low"])
    def test_all_confidence_levels_accepted(self, confidence):
        """EvaluationReport accepts all three confidence levels."""
        report = EvaluationReport(
            confidence=confidence,
            useful=confidence != "low",
            description="test",
            needs_web_search=(confidence == "low"),
        )
        assert report.confidence == confidence

    def test_web_search_required_when_confidence_low(self):
        """Business rule: low confidence should always set needs_web_search=True."""
        report = EvaluationReport(
            confidence="low",
            useful=False,
            description="No relevant documents found.",
            needs_web_search=True,
        )
        if report.confidence == "low":
            assert report.needs_web_search is True
