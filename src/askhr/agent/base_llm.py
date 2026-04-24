from askhr.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI

settings = get_settings()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=settings.google_api_key.get_secret_value()
    )

RETRY_KWARGS = {'stop_after_attempt': 3, 'wait_exponential_jitter': True}