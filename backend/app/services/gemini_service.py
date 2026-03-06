"""
Gemini service — thin wrapper around Google Gemini 2.5 Flash via LangChain.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env at project root
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(os.path.normpath(_env_path))

_API_KEY = os.getenv("GEMINI_API_KEY", "")


from app.core.models import QuiltLayout


def call_gemini(system_prompt: str, user_prompt: str) -> dict:
    """Call Gemini 2.5 Flash and return the parsed JSON response.

    Args:
        system_prompt: System-level instructions (role, rules, format).
        user_prompt:   Per-request instructions (canvas, style, etc.).

    Returns:
        Parsed dict from the LLM JSON response.

    Raises:
        RuntimeError: If the API call itself fails.
    """
    if not _API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to the .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=_API_KEY,
        temperature=0.2,
    )

    structured_llm = llm.with_structured_output(QuiltLayout)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    

    try:
        # Returns a validated QuiltLayout Pydantic object
        parsed_quilt = structured_llm.invoke(messages)
        return parsed_quilt.model_dump()
    except Exception as e:
        raise RuntimeError(f"Gemini Tool Call failed: {e}") from e
