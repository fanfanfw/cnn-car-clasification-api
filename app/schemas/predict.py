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
                "predicted_class": "label_a",
                "confidence": 0.93,
                "probabilities": {
                    "label_a": 0.93,
                    "label_b": 0.05,
                    "label_c": 0.02
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
    hierarchical_loaded: bool = False
    hierarchical_device: str = "not loaded"
    version: str


class HierStageResult(BaseModel):
    predicted: str = Field(..., description="Predicted label for the stage")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for this stage")


class HierPredictionResult(BaseModel):
    final_label: str = Field(..., description="Final predicted label (e.g. X_gen3facelift)")
    final_display: str = Field(..., description="Human friendly label with years")
    needs_review: bool = Field(..., description="True if prediction confidence is low")
    policy_applied: Dict[str, object] = Field(..., description="Policy/threshold application details")
    stage1: HierStageResult
    stage2: HierStageResult


class HierPredictionResponse(BaseModel):
    success: bool = True
    message: str = "Prediction successful"
    data: HierPredictionResult


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
