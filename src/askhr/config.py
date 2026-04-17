from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
   model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

   google_api_key: SecretStr


def get_settings(): 
   return Settings()
