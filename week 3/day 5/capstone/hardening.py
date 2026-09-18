"""
Week 3 Day 5 — System Hardening & Security Guardrails
=====================================================
Provides enterprise-grade hardening for the AFL Assistant:
1. Prompt Injection Defense (detects and neutralizes jailbreaks and scope overrides).
2. Session Abuse & Rate Monitoring (tracks consecutive off-topic / attack probes).
3. Defensive Tool Execution with Timeouts.
"""

import re
import concurrent.futures
from typing import Dict, Any, Optional, Tuple, Callable

# Robust prompt injection and scope override patterns
INJECTION_PATTERNS = [
    r'ignore\b.+(previous|prior|all)\b.+(instruction|prompt|rule)',
    r'disregard\b.+(previous|prior|all)\b.+(instruction|rule)',
    r'system\s+override',
    r'\bdan\b',
    r'you\s+are\s+now\b',
    r'unconstrained',
    r'forget\b.+(you\s+are|afl)',
    r'pretend\b.+(you\s+are|to\s+be)\s+(a\s+)?(general|python|code|math)',
    r'reveal\b.+(system\s+prompt|instruction|prompt)',
    r'bypass\b.+(safety|domain|afl|filter|restriction|paywall)',
    r'act\s+as\s+an\s+unrestricted',
    r'jailbreak',
    r'sudo\s+mode',
    r'developer\s+mode\s+enabled'
]

# In-memory session tracking for abuse / repetitive off-topic probing
_SESSION_ABUSE_TRACKER: Dict[str, int] = {}
ABUSE_THRESHOLD = 3


def detect_prompt_injection(query: str) -> Tuple[bool, Optional[str]]:
    """Analyzes incoming query for prompt injection or instruction override patterns.

    Args:
        query: Incoming user query string.

    Returns:
        (is_injection, reason_description)
    """
    q = query.strip().lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q):
            return True, f"Security Alert: Query matched injection pattern '{pattern}' attempting to override domain scope."
    return False, None


def check_session_abuse(session_id: str, is_off_topic_or_injection: bool) -> Tuple[bool, Optional[str]]:
    """Tracks consecutive off-topic or injection queries per session.

    Args:
        session_id: Session/conversation thread identifier.
        is_off_topic_or_injection: Whether the current turn was classified as off-topic or malicious.

    Returns:
        (is_rate_limited, warning_message)
    """
    count = _SESSION_ABUSE_TRACKER.get(session_id, 0)
    if is_off_topic_or_injection:
        count += 1
        _SESSION_ABUSE_TRACKER[session_id] = count
    else:
        # Reset counter on legitimate AFL query
        _SESSION_ABUSE_TRACKER[session_id] = 0
        return False, None

    if count >= ABUSE_THRESHOLD:
        msg = (
            f"Notice: Multiple consecutive out-of-scope requests ({count}) detected for this session. "
            "This assistant is strictly domain-locked to Australian Rules Football (AFL). "
            "Please submit legitimate AFL inquiries to continue normal operation."
        )
        return True, msg

    return False, None


def reset_session_abuse(session_id: str):
    """Resets abuse counter for a given session."""
    _SESSION_ABUSE_TRACKER.pop(session_id, None)


def execute_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None, timeout_seconds: float = 4.0) -> Any:
    """Executes a callable with a strict execution timeout to prevent pipeline stalling.

    Args:
        func: The tool or worker function to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        timeout_seconds: Maximum allowed runtime in seconds.

    Returns:
        Function result or raises TimeoutError.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Tool execution timed out after {timeout_seconds} seconds.")
