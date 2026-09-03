# POC #13 — Structured Vehicle Symptom Extraction

This standalone internship POC converts an owner's free-form vehicle description into conservative, structured observations. It extracts what the user said; it does not diagnose mechanical causes.

## Architecture

- `symptom_extraction/service.py` — stable reusable entry point.
- `extractor.py` — replaceable extractor interface and deterministic implementation.
- `vocabulary.py` — centralized symptom and operating-condition aliases.
- `negation.py` — negation, uncertainty, and third-party context rules.
- `normalization.py` — text, severity, and bounded-score helpers.
- `evidence.py` — normalized user-report evidence conversion.
- `summaries.py` — deterministic, non-diagnostic summaries.
- `schemas.py` — Pydantic contracts.
- `app.py` — thin FastAPI transport only.

The package has no FastAPI dependency. FastAPI is imported only by `app.py`.

## Reusable interface

```python
from symptom_extraction.service import SymptomExtractionService

service = SymptomExtractionService()
result = service.extract("The car shakes at idle and I hear a ticking noise.")
evidence = result.to_evidence()
```

`SymptomExtractor` defines the replaceable extraction abstraction. A future NLP implementation can implement it without changing the service contract.

## Extraction behavior

The centralized vocabulary covers shaking/vibration, rough idle, common sounds and smells, smoke and leaks, dashboard warnings, temperature/start/stall/power observations, steering/brake/tire behavior, rust/corrosion, and reported oil state. Aliases normalize everyday wording to canonical events without mapping observations to causes.

Operating conditions include cold start, warm engine, idle, acceleration/deceleration, braking, turning, speed/RPM, load, startup/shutdown, after driving, and wet conditions. Temporal rules preserve explicit onset, duration, frequency, and trend phrases. Missing facts remain `null`.

Negated observations remain in `symptoms` with `negated=true` and are excluded from normal positive evidence. `to_evidence(include_negated=True)` emits an explicit `no_<event>` event so it cannot be confused with a positive report. Uncertainty markers reduce extraction confidence and set `uncertain=true` without discarding the observation.

Severity comes only from nearby explicit modifiers such as `small`, `loud`, `badly`, or `violently`. Default severity is neutral and conservative. Extraction confidence means confidence that the user expressed an observation—not the probability that the vehicle has a defect or that any mechanical diagnosis is correct.

Third-party statements such as “my friend said his car…” are ignored. A mechanic checking for a leak and not finding one becomes a negated observation rather than positive leak evidence.

## Evidence Fusion compatibility

Positive observations produce the existing normalized shape:

```json
{
  "source": "user_report",
  "event": "vehicle_shaking",
  "severity": 0.7,
  "confidence": 0.92,
  "explanation": "User reported vehicle shaking under idle conditions.",
  "metadata": {
    "conditions": ["idle"],
    "original_phrase": "car shakes",
    "uncertain": false,
    "negated": false
  }
}
```

This POC does not import or call Evidence Fusion.

## API

`GET /health`

`POST /api/v1/diagnostics/symptoms`

```json
{ "text": "The car shakes at idle and the engine light is on." }
```

The response is a `SymptomExtractionResult` containing original text, symptoms, conditions, temporal information, evidence, summary, and limitations. No authentication, database, or external service is required.

## Install, run, and test

From this directory:

```powershell
python -m pip install -r requirements.txt
python -m compileall .
pytest -q
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8013
```

Open `http://127.0.0.1:8013/docs`. Sample descriptions are in `samples/sample_descriptions.json`.

Tests cover aliases, multiple symptoms, duplicate suppression, negation, uncertainty, severity, conditions, temporal information, trends, evidence bounds, empty/long input, third-party reports, mechanic negative observations, deterministic serialization, summaries, and both HTTP routes.

## Limitations

- Rules can miss uncommon expressions, spelling errors, long-distance references, and complex grammar.
- Negation and modifier scope are approximate character/sentence windows.
- Conditions currently apply as result-level metadata rather than being linked to individual symptoms in multi-clause descriptions.
- Generic and contextual events can both appear (for example, a brake squeal can produce a squealing sound and brake noise).
- English is the only supported language.
- User statements are unverified and this POC provides no repair or safety diagnosis.

## Future backend integration

The future AutoAssist backend can import `service.py` plus the package modules it uses: `schemas.py`, `extractor.py`, `vocabulary.py`, `normalization.py`, `negation.py`, `evidence.py`, and `summaries.py`, along with the centralized values in `config.py`. `app.py` should remain standalone transport and should not be imported into the shared backend.
