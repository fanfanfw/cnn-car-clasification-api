from fastapi import APIRouter

from app.config import get_settings
from app.schemas.predict import HealthResponse
from app.services.classifier import classifier

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API and model status"
)
async def health_check():
    settings = get_settings()
    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        model_loaded=classifier.is_loaded,
        device=classifier.device_name if classifier.is_loaded else "not loaded",
        version=settings.api_version
    )
