# src/config.py
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "RAG System"
    ENVIRONMENT: str = "development"
    
    # Path settings
    PDF_PATH: str = os.path.join("data", "Africa_Developer_Ecosystem_Report_2021.pdf")
    
    # Model settings
    MODEL_NAME: str = "gemini-pro"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1024
    
    class Config:
        env_file = ".env.dev"

@lru_cache()
def get_settings():
    return Settings()