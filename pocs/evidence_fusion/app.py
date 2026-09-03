"""Thin FastAPI adapter for the evidence-fusion service."""

from fastapi import FastAPI

from evidence_fusion.schemas import FusionRequest, FusionResult
from evidence_fusion.service import EvidenceFusionService


app = FastAPI(title="Evidence Fusion POC", version="0.1.0")
service = EvidenceFusionService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "evidence-fusion-poc", "engine": "deterministic-rules"}


@app.post("/api/v1/diagnostics/fuse", response_model=FusionResult)
def fuse_evidence(request: FusionRequest) -> FusionResult:
    return service.fuse(request.evidence, request.user_description)
