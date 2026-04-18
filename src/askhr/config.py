from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
   model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

   google_api_key: SecretStr

   pinecone_api_key: SecretStr
   pinecone_index_name: str = "askhr"
   pinecone_cloud: str = "aws"
   pinecone_region: str = "us-east-1"
   embedding_dimension: int = 768

def get_settings(): 
   return Settings()
