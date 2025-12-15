from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    predicted_class: str = Field(..., description="Predicted car class (model_variant)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    tta_enabled: bool = Field(default=False, description="Whether TTA was used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "vios_g",
                "confidence": 0.9523,
                "probabilities": {
                    "vios_g": 0.9523,
                    "alphard_sc": 0.0312,
                    "harrier_z": 0.0165
                },
                "tta_enabled": False
            }
        }


class PredictionResponse(BaseModel):
    success: bool = True
    message: str = "Prediction successful"
    data: PredictionResult


class BatchPredictionItem(BaseModel):
    filename: str
    prediction: Optional[PredictionResult] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    success: bool = True
    message: str = "Batch prediction completed"
    total: int
    successful: int
    failed: int
    data: List[BatchPredictionItem]


class ClassesResponse(BaseModel):
    success: bool = True
    total: int
    classes: List[str]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    device: str
    version: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Prediction failed",
                "detail": "Invalid image format"
            }
        }
