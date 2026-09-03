# Intelligent Evidence Fusion POC

This internship POC combines already-structured findings from vehicle image, audio, video, user-report, manual, and telemetry sources. It groups duplicate evidence, identifies limited explicit contradictions, ranks a small set of troubleshooting hypotheses, and produces a deterministic summary.

It does not diagnose a vehicle. Fusion scores are explainable POC ranking values, not calibrated probabilities.

## Architecture

~~~text
FastAPI app.py
    -> EvidenceFusionService
        -> normalization and image adapter
        -> same-source/event grouping
        -> explicit conflict rules
        -> hypothesis rules and scoring
        -> deterministic summary
        -> Pydantic result
~~~

All fusion behavior is in the evidence_fusion package. It has no FastAPI, Streamlit, LLM, RAG, or mobile dependency.

## Evidence contract

Required:

~~~json
{
  "source": "video",
  "event": "possible_engine_vibration",
  "severity": 0.61,
  "confidence": 0.72
}
~~~

Both scores must be within 0–1. Optional fields are timestamp, metadata, explanation, and origin_id. Supported source names include image, audio, video, user_report, manual, and telemetry; an unknown source receives the centralized fallback weight.

User descriptions remain context only and are not parsed into evidence.

## Source weights

~~~text
image       1.00
audio       0.90
video       0.90
user_report 0.65
manual      0.75
telemetry   1.00
unknown     0.60
~~~

These are assumptions for demonstration and have not been statistically calibrated.

## Image adapter

The existing YOLO report can be adapted without modifying it:

~~~python
from evidence_fusion.normalization import image_detections_to_evidence

evidence = image_detections_to_evidence(
    [{"label": "oil_leak", "confidence": 0.81, "bbox": [10, 20, 80, 90]}],
    origin_id="inspection-1",
)
~~~

Image severity defaults to 0.50 unless explicitly provided.

## Grouping and deduplication

Evidence is canonicalized and grouped by source plus event. The strongest confidence is retained as the primary value. Further duplicates add only 15% of their confidence before the grouped confidence is capped at one. Severity is confidence-weighted.

Consequently, three rust boxes from image evidence form one image/rust group and one independent source, not three confirmations. Corroboration is based on distinct sources supporting the same hypothesis.

## Hypothesis rules

The deliberately small hypothesis set is:

- lubrication_issue: oil leak, low-oil warning, knocking-like audio, structured low-oil report, or weak generic dashboard support.
- engine_running_irregularly: misfire-like audio, engine vibration, structured shaking report, check-engine warning, or weak generic dashboard support.
- cooling_or_combustion_issue: smoke, overheating warning/report, or weak generic dashboard support.
- belt_or_accessory_issue: broken belt, bearing-like noise, weak hissing support, or charging warning.
- tire_condition_issue: tire wear, pressure warning, pulling, or steering vibration.
- corrosion_issue: rust or structured corrosion report.

Aliases are explicit. Generic dashboard_indicator evidence is deliberately weak because the existing image class does not identify which warning is illuminated.

## Scoring

For each supporting grouped item:

~~~text
contribution = confidence × severity × source weight × event-rule weight
~~~

Base score combines contributions with noisy-OR:

~~~text
base = 1 - product(1 - contribution)
~~~

Add 0.08 for each additional independent source, capped at 0.16. Subtract 0.18 for each applicable strong conflict, with the total penalty capped at 0.50. Final values are clamped to 0–1 and reported with a cautious maximum of 0.95 so a POC ranking never implies certainty.

Confidence uses the same noisy-OR approach without severity. Hypothesis severity is the contribution-weighted mean severity. Hypotheses below 0.12 are omitted, and 0.55 is the POC strong-result threshold.

## Conflicts

Only cautious explicit rules are implemented:

- possible_smoke versus no_visible_smoke from different strong sources.
- warning-light evidence versus no_warning_lights from different strong sources.

Both sides remain visible. The supporting hypothesis is penalized rather than discarded. Same-source contradictions are not treated as independent cross-source conflicts.

## Service use

~~~python
from evidence_fusion.service import EvidenceFusionService

result = EvidenceFusionService().fuse(
    evidence=[
        {"source": "audio", "event": "possible_misfire_pattern", "severity": 0.8, "confidence": 0.85},
        {"source": "video", "event": "possible_engine_vibration", "severity": 0.75, "confidence": 0.82},
        {"source": "user_report", "event": "vehicle_shaking", "severity": 0.7, "confidence": 0.9}
    ],
    user_description="The car shakes at idle."
)
~~~

## API

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
~~~

Endpoints:

- GET /health
- POST /api/v1/diagnostics/fuse
- GET /docs

POST accepts JSON with evidence and optional user_description:

~~~powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/diagnostics/fuse -H "Content-Type: application/json" -d "{\"evidence\":[{\"source\":\"image\",\"event\":\"rust\",\"severity\":0.5,\"confidence\":0.8}]}"
~~~

The response contains evidence_count, grouped_evidence, ranked hypotheses, conflicts, summary, and limitations.

## Tests

~~~powershell
pytest -q
python -m compileall -q app.py config.py evidence_fusion tests
~~~

Tests cover validation, aliases, the image adapter, source weights, duplicate grouping, cross-source corroboration, score clamping, all requested hypothesis scenarios, conflict penalties, weak/unrelated evidence, empty input, summaries, orchestration, and serialization.

## Integration direction

- Image: pass current YOLO detection dictionaries through image_detections_to_evidence.
- Audio: pass AudioDiagnosticResult.to_evidence() records.
- Video: pass VideoDiagnosticResult.to_evidence() records.
- User reports: pass structured events from a future symptom-extraction component.
- Manual/RAG: later convert relevant retrieved context into cautious structured manual evidence; this POC does not perform retrieval.
- Telemetry: later provide normalized sensor events through the same schema.

The shared backend should import EvidenceFusionService and the evidence schemas. The POC FastAPI app and sample JSON files do not need to move.

## Limitations

- Weights, mappings, thresholds, bonuses, and penalties are hand-selected POC assumptions.
- Corroborating incorrect upstream detections can amplify a wrong finding.
- Duplicate grouping assumes source/event identifies a modality group; a production system needs stronger observation and device identity.
- Generic warning-light evidence is ambiguous.
- Conflict rules are intentionally sparse.
- User text is not parsed.
- Manual/RAG retrieval is not implemented.
- Scores do not express diagnostic probability or establish mechanical cause.
