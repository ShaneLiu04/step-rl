"""Security utilities for input sanitization and validation."""

from urllib.parse import urlparse


def escape_css_string(value: str) -> str:
    """Escape a string for safe CSS selector interpolation.

    Handles single quotes, double quotes, backslashes, newlines,
    null bytes, and other special CSS characters.
    """
    if not isinstance(value, str):
        return ""
    # Escape backslashes first, then quotes, then control chars
    value = value.replace("\\", "\\\\")
    value = value.replace("'", "\\'")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n ")
    value = value.replace("\r", "\\r ")
    value = value.replace("\x00", "")
    return value


def escape_xpath_string(value: str) -> str:
    """Escape quotes in a string for safe XPath interpolation."""
    if not isinstance(value, str):
        return ""
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", '.', ".join(f"'{p}'" for p in parts) + ")"


def validate_url(url: str, blocked_domains: set, allowed_domains: set) -> bool:
    """
    Validate URL against block/allow lists using proper domain extraction.
    Returns True if URL is allowed, False if blocked.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        return False

    hostname_lower = hostname.lower()

    # Check blocked domains (exact domain or subdomain match)
    for blocked in blocked_domains:
        blocked = blocked.lower().strip()
        if not blocked:
            continue
        if hostname_lower == blocked or hostname_lower.endswith("." + blocked):
            return False

    # Check allowed domains (if specified, only allow exact matches)
    if allowed_domains:
        for allowed in allowed_domains:
            allowed = allowed.lower().strip()
            if not allowed:
                continue
            if hostname_lower == allowed or hostname_lower.endswith("." + allowed):
                return True
        return False  # allowed_domains specified but no match

    return True
