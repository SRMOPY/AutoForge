"""
Input validation for AutoForge.
Validates and sanitizes all user-provided input before it touches any agent.
"""

import re

MIN_DESCRIPTION_LENGTH = 10
MAX_DESCRIPTION_LENGTH = 2000

# Patterns that suggest prompt injection or abuse attempts
SUSPICIOUS_PATTERNS = [
    r"ignore (previous|above|all) instructions",
    r"you are now",
    r"disregard your",
    r"new instructions:",
    r"system prompt",
    r"jailbreak",
]


class ValidationError(ValueError):
    """Raised when user input fails validation."""
    pass


def validate_project_description(description: str) -> str:
    """
    Validate and sanitize a project description.

    Returns the cleaned description if valid.
    Raises ValidationError with a human-friendly message if not.
    """
    if not isinstance(description, str):
        raise ValidationError("Project description must be text.")

    cleaned = description.strip()

    if not cleaned:
        raise ValidationError(
            "Project description cannot be empty.\n"
            "Example: 'a REST API for a todo list with user authentication'"
        )

    if len(cleaned) < MIN_DESCRIPTION_LENGTH:
        raise ValidationError(
            f"Description is too short ({len(cleaned)} chars). "
            f"Please describe your project in at least {MIN_DESCRIPTION_LENGTH} characters."
        )

    if len(cleaned) > MAX_DESCRIPTION_LENGTH:
        raise ValidationError(
            f"Description is too long ({len(cleaned)} chars). "
            f"Please keep it under {MAX_DESCRIPTION_LENGTH} characters."
        )

    lower = cleaned.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lower):
            raise ValidationError(
                "Description contains unsupported content. "
                "Please describe a software project you want to build."
            )

    return cleaned


def validate_provider(provider: str) -> str:
    """Validate the model provider name."""
    supported = ["gemini", "groq", "openai"]
    cleaned = provider.strip().lower()
    if cleaned not in supported:
        raise ValidationError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(supported)}"
        )
    return cleaned
