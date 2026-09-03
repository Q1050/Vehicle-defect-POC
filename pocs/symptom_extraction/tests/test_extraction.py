import pytest


def events(result):
    return {item.event for item in result.symptoms if not item.negated}


def test_basic_and_multiple_symptoms(service):
    result = service.extract("The car shakes at idle, the check engine light is on, and I smell fuel after driving.")
    assert {"vehicle_shaking", "check_engine_warning", "fuel_smell"} <= events(result)
    assert {item.condition for item in result.conditions} >= {"idle", "after_driving"}


def test_no_diagnosis_inference(service):
    assert events(service.extract("My car shakes.")) == {"vehicle_shaking"}
    assert "misfire" not in str(service.extract("My car shakes.").model_dump())


def test_conditions_temporal_and_trends(service):
    result = service.extract("My car shakes badly every morning for about two weeks but gets smoother after it warms up.")
    assert {item.condition for item in result.conditions} >= {"cold_start", "warm_engine"}
    assert result.temporal.duration == "for about two weeks"
    assert result.temporal.frequency == "every morning"
    assert result.temporal.trend == "improves_when_warm"


def test_onset_and_under_load(service):
    result = service.extract("It started yesterday and loses power going uphill.")
    assert result.temporal.onset == "started yesterday"
    assert "under_load" in {item.condition for item in result.conditions}
    assert "loss_of_power" in events(result)


def test_empty_and_long_text(service):
    with pytest.raises(ValueError, match="empty"):
        service.extract("   ")
    with pytest.raises(ValueError, match="exceed"):
        service.extract("x" * 20_001)
