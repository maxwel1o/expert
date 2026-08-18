import re


class UnsafeTextError(ValueError):
    """Raised when progress metadata appears to contain a credential."""


_PATTERNS = (
    re.compile(
        r"(?i)\b(password|passwd|token|api[_ -]?key|secret)\s*[:=]\s*\S{8,}"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)https?://[^/\s:@]+:[^/\s@]+@"),
)


def validate_safe_text(value: str, field: str) -> str:
    cleaned = value.strip()
    if any(pattern.search(cleaned) for pattern in _PATTERNS):
        raise UnsafeTextError(
            f"{field} appears to contain a secret; use a credential alias"
        )
    return cleaned

