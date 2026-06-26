from typing import Any, Dict


DEFAULT_REVIEW_PARAMETERS = {
    "tone": "neutral",
    "depth": "detailed",
    "persona": "scientific_rigor_focus",
}

TONE_GUIDANCE = {
    "constructive": (
        "Use an improvement-oriented voice. Surface weaknesses as actionable "
        "recommendations without softening evidence standards."
    ),
    "critical": (
        "Use a stricter reviewing voice. Emphasize risks, unsupported claims, "
        "and publication-blocking concerns while remaining fair."
    ),
    "neutral": (
        "Use a balanced, objective voice. Weigh strengths and weaknesses evenly "
        "and keep the recommendation evidence-first."
    ),
}

DEPTH_GUIDANCE = {
    "summary": (
        "Be concise. Keep the most decisive strengths, weaknesses, risks, and "
        "recommendation rationale; avoid secondary detail."
    ),
    "detailed": (
        "Use the current reviewer depth. Provide concrete strengths, weaknesses, "
        "risks, and recommendation rationale."
    ),
    "comprehensive": (
        "Provide richer reasoning and more complete coverage within the existing "
        "workflow, claim extraction, and retrieval limits."
    ),
}

PERSONA_GUIDANCE = {
    "editorial_focus": (
        "Prioritize novelty, clarity, structure, audience fit, and publication "
        "suitability. Do not ignore scientific or methodological risks."
    ),
    "methodological_focus": (
        "Prioritize study design, statistics, reproducibility, data quality, "
        "controls, confounding, and validity. Do not ignore publication fit."
    ),
    "scientific_rigor_focus": (
        "Prioritize scientific correctness, evidence grounding, theoretical "
        "soundness, and unsupported or contradicted claims."
    ),
}


def _valid_or_default(value: Any, allowed: Dict[str, str], default: str) -> str:
    return value if value in allowed else default


def normalize_review_parameters(value: Any) -> Dict[str, str]:
    if value is None:
        return dict(DEFAULT_REVIEW_PARAMETERS)

    if hasattr(value, "model_dump"):
        raw = value.model_dump()
    elif isinstance(value, dict):
        raw = value
    else:
        raw = {}

    return {
        "tone": _valid_or_default(
            raw.get("tone"),
            TONE_GUIDANCE,
            DEFAULT_REVIEW_PARAMETERS["tone"],
        ),
        "depth": _valid_or_default(
            raw.get("depth"),
            DEPTH_GUIDANCE,
            DEFAULT_REVIEW_PARAMETERS["depth"],
        ),
        "persona": _valid_or_default(
            raw.get("persona"),
            PERSONA_GUIDANCE,
            DEFAULT_REVIEW_PARAMETERS["persona"],
        ),
    }


def build_review_parameter_guidance(value: Any) -> str:
    params = normalize_review_parameters(value)

    tone = params["tone"]
    depth = params["depth"]
    persona = params["persona"]

    return (
        "Review parameter guidance:\n"
        f"- Tone ({tone}): {TONE_GUIDANCE[tone]}\n"
        f"- Depth ({depth}): {DEPTH_GUIDANCE[depth]}\n"
        f"- Persona ({persona}): {PERSONA_GUIDANCE[persona]}\n"
        "- Apply these settings as emphasis controls only. Do not change the "
        "required JSON schema, do not add unsupported claims, and do not lower "
        "the evidence threshold."
    )
