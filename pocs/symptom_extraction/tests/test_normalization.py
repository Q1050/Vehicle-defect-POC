def positive(result):
    return [item for item in result.symptoms if not item.negated]


def test_alias_normalization_and_duplicate_suppression(service):
    result = service.extract("The car shakes and the car vibrates badly.")
    shaking = [item for item in positive(result) if item.event == "vehicle_shaking"]
    assert len(shaking) == 1
    assert shaking[0].severity == 0.7


def test_warning_and_sound_aliases(service):
    result = service.extract("CEL is on and I hear a tick tick noise.")
    assert {item.event for item in positive(result)} >= {"check_engine_warning", "ticking_sound"}


def test_deterministic_serialization(service):
    text = "The tire pressure light is on and the front tire looks worn."
    assert service.extract(text).model_dump_json() == service.extract(text).model_dump_json()
