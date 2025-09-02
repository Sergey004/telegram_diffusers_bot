import os
import asyncio
from typing import Optional

from openai import OpenAI
from google.colab import userdata

# Initialize the OpenAI client for Nvidia NIM (if API key is provided)
_nvidia_api_key: Optional[str] = os.getenv("NVIDIA_API_KEY") or userdata.get('NVIDIA_API_KEY')
_client: Optional[OpenAI] = None
if _nvidia_api_key:
    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=_nvidia_api_key,
    )

async def generate_prompt_from_idea(idea_text: str) -> str:
    """
    Sends the user‑provided idea text to the Nvidia NIM LLM and returns a
    tag‑based prompt suitable for Stable Diffusion.

    If the NVIDIA_API_KEY is not set, the function falls back to returning the
    original text unchanged (so the bot can still operate without LLM support).
    """
    if not _client:
        # No LLM available – return the raw idea as a prompt
        return idea_text.strip()

    # The request is synchronous; run it in a thread to avoid blocking the event loop
    def _call_llm() -> str:
        response = _client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": idea_text}],
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=False,
        )
        # The content may contain newline characters – strip excess whitespace
        return response.choices[0].message.content.strip()

    return await asyncio.to_thread(_call_llm)
