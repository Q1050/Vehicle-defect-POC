def test_summary_is_observational(service):
    result = service.extract("The car shakes at idle and I hear ticking during a cold start.")
    assert "reported" in result.summary
    assert "not been mechanically verified" in result.summary
    assert not any(term in result.summary.lower() for term in ("spark plug", "engine mount", "injector"))


def test_no_supported_symptom_summary(service):
    result = service.extract("It has been getting worse for two weeks.")
    assert not result.symptoms
    assert result.temporal.duration == "for two weeks"
    assert result.temporal.trend == "worsening"
    assert "No supported vehicle symptoms" in result.summary
