from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import health, predict
from app.services.classifier import classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    print(f"Loading model from {settings.model_path}...")
    try:
        classifier.load_model()
        print(f"Model loaded successfully on {classifier.device_name}")
        print(f"Available classes: {classifier.classes}")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
## Car Classification API

REST API for classifying car images.

### Features
- Single image prediction
- Batch prediction (up to 10 images)
- Test Time Augmentation (TTA) for improved accuracy
- API Key authentication
- Model classes are loaded from the checkpoint; use `/api/v1/classes` to see what the current model supports

### Authentication
Include `X-API-Key` header in all requests to `/api/v1/*` endpoints.

### Supported Image Formats
- JPEG
- PNG  
- WebP
- BMP
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(health.router)
    app.include_router(predict.router)
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Internal server error",
                "detail": str(exc) if settings.debug else None
            }
        )
    
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


app = create_app()
