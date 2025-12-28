from fastapi import APIRouter

from app.config import get_settings
from app.schemas.predict import HealthResponse
from app.services.classifier import classifier
from app.services.hier_classifier import hier_classifier

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and model status"
)
async def health_check():
    settings = get_settings()
    overall_ok = classifier.is_loaded or hier_classifier.is_loaded
    return HealthResponse(
        status="healthy" if overall_ok else "degraded",
        model_loaded=classifier.is_loaded,
        device=classifier.device_name if classifier.is_loaded else "not loaded",
        hierarchical_loaded=hier_classifier.is_loaded,
        hierarchical_device=hier_classifier.device_name if hier_classifier.is_loaded else "not loaded",
        version=settings.api_version
    )
