"""Shared Gemini chat model for VAANI's title, summary, extraction and RAG chat.

The key is read from the GEMINI_API_KEY environment variable. app.py fills the environment from
.env locally and from Streamlit secrets on Streamlit Community Cloud. The key is never printed or
logged, and it is not part of any error message produced here.
"""
import os

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.errors import UserFacingError

# A free-tier "Flash-Lite" model (fast, cheap tier for summarising/extraction). Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT = 60   # seconds per Gemini request
MAX_ATTEMPTS = 2       # 1 retry: free-tier quota errors are not retried aggressively


class LLMResponseError(UserFacingError):
    """Gemini answered, but without usable text (empty or blocked response)."""


def gemini_model_name() -> str:
    return (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()


def get_gemini_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise UserFacingError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to your .env file (local) "
            "or to the app's Secrets on Streamlit Cloud."
        )
    return key


def _to_text_message(message):
    """Normalise Gemini's reply to a plain-text AIMessage and reject empty/blocked answers."""
    content = getattr(message, "content", None)
    if isinstance(content, list):  # some models return a list of content parts
        content = "".join(p if isinstance(p, str) else str(p.get("text", "")) if isinstance(p, dict) else ""
                          for p in content)
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("Gemini returned an empty response (it may have been blocked). Please try again.")
    return AIMessage(content=content)


def get_llm(temperature: float):
    """Gemini chat model used in VAANI's LangChain chains (prompt | get_llm(...) | StrOutputParser())."""
    llm = ChatGoogleGenerativeAI(
        model=gemini_model_name(),
        google_api_key=get_gemini_api_key(),
        temperature=temperature,
        max_retries=MAX_ATTEMPTS,
        timeout=REQUEST_TIMEOUT,
    )
    return llm | RunnableLambda(_to_text_message)


def describe_llm_error(error: Exception) -> str:
    """A readable message for a failed Gemini call (never includes the key or raw request details)."""
    if isinstance(error, UserFacingError):
        return str(error)
    name, text = type(error).__name__, str(error)
    lowered = text.lower()
    if name == "ResourceExhausted" or "429" in text or "quota" in lowered or "resource_exhausted" in lowered:
        return ("The Gemini free-tier quota or rate limit was reached. Wait a minute and try again; "
                "daily limits reset at midnight Pacific time.")
    if "api key not valid" in lowered or "api_key_invalid" in lowered or name in ("PermissionDenied", "Unauthenticated"):
        return "The Gemini API key was rejected. Check GEMINI_API_KEY in .env (local) or Streamlit secrets (cloud)."
    if name in ("DeadlineExceeded", "Timeout", "ReadTimeout") or "timed out" in lowered or "deadline" in lowered:
        return "The Gemini request timed out. Please try again."
    if name in ("ServiceUnavailable", "InternalServerError") or "503" in text or "500" in text:
        return "Gemini is temporarily unavailable. Please try again in a moment."
    if "model" in lowered and ("not found" in lowered or "404" in text):
        return "The configured Gemini model is not available for this API key (check GEMINI_MODEL)."
    return f"The Gemini request failed ({name}). Please try again."
