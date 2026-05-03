"""UC3M LLM integration module using Ollama API."""

import requests
import urllib3

from config import LLM_API_URL, LLM_API_KEY, DEFAULT_MODEL, DEFAULT_TEMPERATURE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def generate(prompt: str, system_prompt: str = '', model: str = None, temperature: float = None) -> str:
    """Generate text using the UC3M Ollama API."""
    model = model or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE

    full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            LLM_API_URL,
            json=payload,
            headers={"X-API-KEY": LLM_API_KEY},
            verify=False,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.Timeout:
        return "[Error] Request timed out after 60s."
    except requests.exceptions.ConnectionError:
        return "[Error] Could not connect to LLM API."
    except requests.exceptions.HTTPError as e:
        return f"[Error] API returned status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"[Error] {e}"
