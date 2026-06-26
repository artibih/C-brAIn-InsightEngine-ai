import re
import json

def extract_json(text: str) -> dict:
    """
    Extract the first valid JSON object from an LLM response.
    Handles markdown fences and trailing explanations.
    """

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    json_str = match.group(0)


    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON object in LLM response") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected top-level JSON object")
    return payload