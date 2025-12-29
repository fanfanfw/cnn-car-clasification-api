from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.dependencies import verify_api_key
from app.routers.predict import MAX_FILE_SIZE, validate_image
from app.schemas.predict import HierPredictionResponse, HierPredictionResult, HierStageResult
from app.services.hier_classifier import hier_classifier


router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction (ALPHARD SC/SA/X)"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/predict/hier",
    response_model=HierPredictionResponse,
    summary="Predict Car Label (Hierarchical 2-stage)",
    description="Upload an image to predict variant first, then generation per-variant.",
)
async def predict_hier(
    file: UploadFile = File(..., description="Car image file (JPEG, PNG, WebP)"),
    allow_needs_review: bool = Query(True, description="If false, returns 422 when prediction needs review."),
):
    validate_image(file)
    if not hier_classifier.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Hierarchical models not loaded")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
        )

    result = hier_classifier.predict(contents)
    if result["needs_review"] and not allow_needs_review:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Prediction needs review")

    return HierPredictionResponse(
        success=True,
        message="Prediction successful",
        data=HierPredictionResult(
            final_label=result["final_label"],
            final_display=result["final_display"],
            needs_review=bool(result["needs_review"]),
            policy_applied=result["policy_applied"],
            stage1=HierStageResult(**result["stage1"]),
            stage2=HierStageResult(**result["stage2"]),
        ),
    )


@router.post(
    "/predict/hier/batch",
    response_model=List[HierPredictionResult],
    summary="Batch Predict (Hierarchical)",
    description="Batch hierarchical prediction (max 10 images).",
)
async def predict_hier_batch(
    files: List[UploadFile] = File(..., description="Car image files (max 10)"),
    allow_needs_review: bool = Query(True, description="If false, items needing review return error entries."),
):
    if not hier_classifier.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Hierarchical models not loaded")
    if len(files) > 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 10 images per batch")

    results: List[HierPredictionResult] = []
    for file in files:
        validate_image(file)
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        item = hier_classifier.predict(contents)
        if item["needs_review"] and not allow_needs_review:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Prediction needs review")
        results.append(
            HierPredictionResult(
                final_label=item["final_label"],
                final_display=item["final_display"],
                needs_review=bool(item["needs_review"]),
                policy_applied=item["policy_applied"],
                stage1=HierStageResult(**item["stage1"]),
                stage2=HierStageResult(**item["stage2"]),
            )
        )

    return results

