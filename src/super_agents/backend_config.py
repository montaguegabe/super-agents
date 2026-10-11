"""Single source of truth for Super Agents coding-backend selection.

Backend constants, the alias map, ``.env`` parsing, and environment-based
backend resolution live here. ``defaults``, ``backend_clients``, and
``backend`` re-export these names for compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path

CODEX_BACKEND = "codex"
OPENBASE_CLOUD_BACKEND = "openbase_cloud"
OPENBASE_CLOUD_CODEX_BACKEND = "openbase_cloud_codex"
CLAUDE_CODE_BACKEND = "claude_code"
BACKENDS = {
    CODEX_BACKEND,
    OPENBASE_CLOUD_BACKEND,
    CLAUDE_CODE_BACKEND,
    OPENBASE_CLOUD_CODEX_BACKEND,
}
CODEX_COMPATIBLE_BACKENDS = {CODEX_BACKEND, OPENBASE_CLOUD_CODEX_BACKEND}
CODING_BACKEND_ENV_KEY = "OPENBASE_CODING_BACKEND"
DEFAULT_BACKEND_ENV_KEY = "SUPER_AGENTS_DEFAULT_BACKEND"
DEFAULT_ENV_FILE = Path.home() / ".openbase" / ".env"
BACKEND_ALIASES = {
    "": CODEX_BACKEND,
    "codex": CODEX_BACKEND,
    "codecs": CODEX_BACKEND,
    "openbase cloud": OPENBASE_CLOUD_BACKEND,
    "claude code": CLAUDE_CODE_BACKEND,
    "cloud code": CLAUDE_CODE_BACKEND,
    "openbase cloud codex": OPENBASE_CLOUD_CODEX_BACKEND,
    "openbase cloud codecs": OPENBASE_CLOUD_CODEX_BACKEND,
    "codex via openbase cloud": OPENBASE_CLOUD_CODEX_BACKEND,
}


def normalize_backend(value: str | None) -> str:
    raw = " ".join((value or "").strip().lower().replace("_", " ").replace("-", " ").split())
    try:
        return BACKEND_ALIASES[raw]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unsupported {CODING_BACKEND_ENV_KEY}: {value}. Supported backends: {supported}.") from exc


def execution_backend(backend: str) -> str:
    if backend in CODEX_COMPATIBLE_BACKENDS:
        return CODEX_BACKEND
    if backend == OPENBASE_CLOUD_BACKEND:
        return CLAUDE_CODE_BACKEND
    return backend


CLAUDE_MODEL_ALIASES = {"fable", "opus", "sonnet", "haiku"}
_CODEX_MODEL_PREFIXES = ("gpt", "o1", "o3", "o4", "codex")

# Canonical model slugs per execution backend. Model choice is the primary
# user-facing abstraction: callers name a model and the backend is inferred
# from this catalog, with ``backend`` kept only as an advanced override.
MODEL_CATALOG: dict[str, set[str]] = {
    CODEX_BACKEND: {"gpt-5.5", "gpt-5", "sol", "astra"},
    CLAUDE_CODE_BACKEND: {"fable", "opus", "sonnet", "haiku"},
}

# Provider-style aliases only (the forms providers themselves publish, plus
# provider-prefixed forms such as ``openai-sol``). Deliberately no phonetic
# or fuzzy aliases: a misheard slug like "seoul" must fail with suggestions,
# not silently start a turn on a model that does not exist.
MODEL_SLUG_ALIASES: dict[str, str] = {
    "openai-sol": "sol",
    "open-ai-sol": "sol",
    "sol-latest": "sol",
    "openai-astra": "astra",
    "open-ai-astra": "astra",
    "astra-latest": "astra",
    "openai-gpt-5.5": "gpt-5.5",
    "gpt5.5": "gpt-5.5",
    "openai-gpt-5": "gpt-5",
    "gpt5": "gpt-5",
    "anthropic-fable": "fable",
    "claude-fable": "fable",
    "claude-fable-5": "fable",
    "fable-5": "fable",
    "anthropic-opus": "opus",
    "claude-opus": "opus",
    "claude-opus-4-8": "opus",
    "anthropic-sonnet": "sonnet",
    "claude-sonnet": "sonnet",
    "claude-sonnet-5": "sonnet",
    "anthropic-haiku": "haiku",
    "claude-haiku": "haiku",
    "claude-haiku-4-5": "haiku",
}

EXTRA_MODELS_ENV_KEY = "SUPER_AGENTS_EXTRA_MODELS"


class UnknownModelError(ValueError):
    """A requested model slug matches no known model, alias, or provider prefix."""


def _normalize_model_slug(value: str) -> str:
    slug = "-".join(value.strip().lower().replace("/", "-").replace("_", "-").replace(" ", "-").split("-"))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _catalog_with_extras() -> dict[str, set[str]]:
    raw = os.environ.get(EXTRA_MODELS_ENV_KEY, "").strip()
    if not raw:
        return MODEL_CATALOG
    import json

    try:
        extras = json.loads(raw)
    except json.JSONDecodeError:
        return MODEL_CATALOG
    if not isinstance(extras, dict):
        return MODEL_CATALOG
    catalog = {backend: set(models) for backend, models in MODEL_CATALOG.items()}
    for backend, models in extras.items():
        if backend in catalog and isinstance(models, list):
            catalog[backend].update(str(model) for model in models if str(model).strip())
    return catalog


def resolve_model(model: str | None) -> tuple[str | None, str | None]:
    """Resolve a model value to ``(canonical_slug, execution_backend)``.

    Resolution order: exact catalog slug, provider-style alias, then a
    provider prefix (``claude*`` / ``gpt*``/``o1``/``o3``/``o4``/``codex*``)
    passed through verbatim for models newer than this catalog. A bare slug
    that matches none of those raises :class:`UnknownModelError` naming the
    valid slugs so callers fail before a turn ever starts.
    """
    if not model or not model.strip():
        return None, None
    slug = _normalize_model_slug(model)
    catalog = _catalog_with_extras()
    for backend, models in catalog.items():
        if slug in models:
            return slug, backend
    aliased = MODEL_SLUG_ALIASES.get(slug)
    if aliased:
        for backend, models in catalog.items():
            if aliased in models:
                return aliased, backend
    if slug.startswith("claude"):
        return model.strip(), CLAUDE_CODE_BACKEND
    if slug.startswith(("anthropic-", "anthropic.")):
        return slug.split("-", 1)[1] if "-" in slug else model.strip(), CLAUDE_CODE_BACKEND
    if slug.startswith(_CODEX_MODEL_PREFIXES):
        return model.strip(), CODEX_BACKEND
    if slug.startswith(("openai-", "open-ai-")):
        remainder = slug.removeprefix("open-ai-").removeprefix("openai-")
        return remainder or model.strip(), CODEX_BACKEND
    known = sorted(slug for models in catalog.values() for slug in models)
    suggestions = _close_model_matches(slug, known)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise UnknownModelError(
        f"Unknown model {model!r}. Known models: {', '.join(known)}. "
        f"Provider-prefixed ids (e.g. openai-sol, claude-fable-5) are also accepted.{hint}"
    )


def _close_model_matches(slug: str, known: list[str]) -> list[str]:
    import difflib

    return difflib.get_close_matches(slug, known, n=3, cutoff=0.6)


def execution_backend_for_model(model: str | None) -> str | None:
    """Which execution backend a model id/alias belongs to, or None if unknown.

    Lets thread creation honor an explicit model choice (e.g. the dispatcher
    asked for "fable") by routing to a backend that can actually run it,
    instead of handing a Claude model to Codex and failing the turn. Unknown
    slugs return None here; use :func:`resolve_model` to reject them.
    """
    if not model:
        return None
    try:
        _canonical, backend = resolve_model(model)
    except UnknownModelError:
        return None
    return backend


def configured_backend_from_environment() -> str:
    env_values = read_env_values(DEFAULT_ENV_FILE)
    return normalize_backend(os.environ.get(CODING_BACKEND_ENV_KEY) or env_values.get(CODING_BACKEND_ENV_KEY))


def default_backend_from_environment() -> str:
    """Return the configured identity used for new threads in this process."""
    override = os.environ.get(DEFAULT_BACKEND_ENV_KEY, "").strip()
    return normalize_backend(override) if override else configured_backend_from_environment()


def backend_from_environment() -> str:
    return execution_backend(configured_backend_from_environment())


def read_env_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = _unquote_env_value(value.strip())
    return values


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
