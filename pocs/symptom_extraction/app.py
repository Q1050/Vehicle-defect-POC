"""Thin FastAPI transport for the reusable extraction service."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from symptom_extraction.schemas import SymptomExtractionResult
from symptom_extraction.service import SymptomExtractionService

app = FastAPI(title="Structured Vehicle Symptom Extraction POC", version="0.1.0")
service = SymptomExtractionService()


class SymptomRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "symptom-extraction-poc", "engine": "deterministic-rules"}


@app.post("/api/v1/diagnostics/symptoms", response_model=SymptomExtractionResult)
def extract_symptoms(request: SymptomRequest) -> SymptomExtractionResult:
    try:
        return service.extract(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
