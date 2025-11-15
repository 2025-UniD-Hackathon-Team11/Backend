
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Backend API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Backend API built with FastAPI"

    # CORS Settings
    ALLOWED_ORIGINS: list[str] = ["*"]

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Database Settings (example - uncomment when needed)
    # DATABASE_URL: Optional[str] = None


settings = Settings()
