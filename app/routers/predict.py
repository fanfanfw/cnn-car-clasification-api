from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.dependencies import verify_api_key
from app.schemas.predict import (
    BatchPredictionItem,
    BatchPredictionResponse,
    ClassesResponse,
    PredictionResponse,
    PredictionResult,
)
from app.services.classifier import classifier

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)]
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_image(file: UploadFile) -> None:
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    
    if file.content_type not in ALLOWED_CONTENT_TYPES and f".{ext}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: JPEG, PNG, WebP, BMP"
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Car Class",
    description="Upload an image to predict the car model/variant"
)
async def predict(
    file: UploadFile = File(..., description="Car image file (JPEG, PNG, WebP)"),
    tta: bool = Query(False, description="Enable Test Time Augmentation for better accuracy (slower)")
):
    validate_image(file)
    
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        predicted_class, confidence, probabilities = classifier.predict(contents, use_tta=tta)
        
        return PredictionResponse(
            success=True,
            message="Prediction successful",
            data=PredictionResult(
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                probabilities=probabilities,
                tta_enabled=tta
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict Car Classes",
    description="Upload multiple images for batch prediction (max 10 images)"
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Car image files (max 10)"),
    tta: bool = Query(False, description="Enable Test Time Augmentation")
):
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch"
        )
    
    results: List[BatchPredictionItem] = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            validate_image(file)
            contents = await file.read()
            
            if len(contents) > MAX_FILE_SIZE:
                raise ValueError(f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")
            
            predicted_class, confidence, probabilities = classifier.predict(contents, use_tta=tta)
            
            results.append(BatchPredictionItem(
                filename=file.filename or "unknown",
                prediction=PredictionResult(
                    predicted_class=predicted_class,
                    confidence=round(confidence, 4),
                    probabilities=probabilities,
                    tta_enabled=tta
                )
            ))
            successful += 1
        except Exception as e:
            results.append(BatchPredictionItem(
                filename=file.filename or "unknown",
                error=str(e)
            ))
            failed += 1
    
    return BatchPredictionResponse(
        success=failed == 0,
        message="Batch prediction completed" if failed == 0 else f"Completed with {failed} error(s)",
        total=len(files),
        successful=successful,
        failed=failed,
        data=results
    )


@router.get(
    "/classes",
    response_model=ClassesResponse,
    summary="Get Available Classes",
    description="Get list of all car classes the model can predict"
)
async def get_classes():
    return ClassesResponse(
        success=True,
        total=len(classifier.classes),
        classes=sorted(classifier.classes)
    )
