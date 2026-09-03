def test_positive_evidence_contract_and_bounds(service):
    result = service.extract("The car shakes violently at idle.")
    evidence = result.to_evidence()[0]
    assert evidence.source == "user_report"
    assert evidence.event == "vehicle_shaking"
    assert evidence.severity == 0.95
    assert 0 <= evidence.confidence <= 1
    assert 0 <= evidence.severity <= 1
    assert evidence.metadata["conditions"] == ["idle"]


def test_uncertain_observation_reduces_confidence(service):
    certain = service.extract("I hear knocking.").symptoms[0]
    uncertain = service.extract("I think I might hear knocking.").symptoms[0]
    assert uncertain.uncertain
    assert uncertain.confidence < certain.confidence
    assert service.extract("I think I might hear knocking.").evidence[0].metadata["uncertain"]
