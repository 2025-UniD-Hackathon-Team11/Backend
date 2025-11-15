from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Backend API",
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }
