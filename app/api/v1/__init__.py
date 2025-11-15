from fastapi import APIRouter

from app.api.v1 import health

api_router = APIRouter()

# Include routers
api_router.include_router(health.router, tags=["health"])

# Add more routers here as needed
# api_router.include_router(users.router, prefix="/users", tags=["users"])
