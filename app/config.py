from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = "changeme"
    api_title: str = "Car Classification API"
    api_version: str = "1.0.0"
    
    model_path: str = "models/best_model.pt"
    img_size: int = 224
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
