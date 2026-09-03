def test_negated_smoke_and_overheating(service):
    result = service.extract("There is no smoke and the engine isn't overheating.")
    found = {item.event: item for item in result.symptoms}
    assert found["visible_smoke"].negated
    assert found["engine_overheating"].negated
    assert not result.evidence


def test_dont_hear_knocking_is_not_positive(service):
    result = service.extract("I don't hear any knocking.")
    assert result.symptoms[0].event == "knocking_sound"
    assert result.symptoms[0].negated
    assert not result.to_evidence()
    assert result.to_evidence(include_negated=True)[0].event == "no_knocking_sound"


def test_third_party_statement_is_ignored(service):
    result = service.extract("My friend said his car makes a knocking noise.")
    assert not result.symptoms


def test_mechanic_negative_observation(service):
    result = service.extract("The mechanic checked for an oil leak but didn't find one.")
    assert result.symptoms[0].event == "oil_leak_reported"
    assert result.symptoms[0].negated
    assert not result.evidence
