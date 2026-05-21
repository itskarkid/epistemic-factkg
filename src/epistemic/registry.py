"""Source trust registry — load and resolve source trustworthiness.

The registry maps source_id → source trust (ST_i) used in the EC formula.
"""

import json
from pathlib import Path


# Fallback ST when source_id is not in registry
DEFAULT_SOURCE_TRUST: float = 0.40

# Known social-media domains for TLD heuristic fallback
_SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset(
    {
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "reddit.com",
        "tiktok.com",
        "linkedin.com",
        "youtube.com",
        "t.co",
        "fb.com",
    }
)

# Compound TLDs that must be preserved as a unit (e.g. bbc.co.uk → bbc, not bbc_co)
_COMPOUND_TLDS: frozenset[str] = frozenset(
    {
        "co.uk",
        "gov.uk",
        "ac.uk",
        "org.uk",
        "me.uk",
        "net.uk",
        "co.in",
        "gov.in",
        "ac.in",
        "co.za",
        "gov.za",
        "ac.za",
        "gov.au",
        "com.au",
        "edu.au",
        "net.au",
        "org.au",
        "gov.ng",
        "gov.gh",
        "gov.ke",
        "gouv.fr",
    }
)


def load_source_trust_registry(path: str | Path) -> dict[str, dict]:
    """Load source_trust_registry.jsonl into a dict keyed by source_id.

    Raises ValueError if any required generic fallback entry is missing.
    Raises ValueError listing missing source_ids if coverage is incomplete.

    Returns:
        {source_id: full registry record dict}
    """
    registry: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                registry[record["source_id"]] = record

    missing = verify_generic_fallback_coverage(registry)
    if missing:
        raise ValueError(
            f"Registry at '{path}' is missing {len(missing)} generic fallback "
            f"entries: {missing}. Add a '{{source_type}}_{{modality}}' entry for each."
        )

    return registry


def make_source_id(domain: str, modality: str) -> str:
    """Build the candidate registry key for a domain + modality pair (no registry needed).

    Strips TLD so make_source_id("reuters.com", "web_text") → "reuters_web_text".
    Use resolve_source_id for full fallback resolution including TLD heuristics.
    """
    domain = domain.lower().removeprefix("www.")
    return f"{_normalise_domain(domain)}_{modality}"


def resolve_source_id(domain: str, modality: str, registry: dict[str, dict]) -> str:
    """Resolve a domain + modality pair to a registry source_id.

    Lookup order:
      1. Exact match:      {domain}_{modality}
      2. Parent domain:    strip first subdomain, retry exact
      3. Archive extract:  web.archive.org — parse embedded original URL, recurse
      4. TLD heuristic:    *.gov, *.edu, *.org
      5. Social media:     known social-media domains
      6. Modality default: web_table → general_web_table, pdf → academic_pdf
      7. Final fallback:   unknown_web
    """
    domain = domain.lower().removeprefix("www.")

    # 1. Exact match
    key = f"{_normalise_domain(domain)}_{modality}"
    if key in registry:
        return key

    # 2. Parent domain (strip first subdomain component)
    parent = _strip_subdomain(domain)
    if parent != domain:
        key = f"{_normalise_domain(parent)}_{modality}"
        if key in registry:
            return key

    # 3. Archive: original URL is embedded in web.archive.org paths
    if domain == "web.archive.org":
        return "webarchive_web_text"

    # 4. TLD heuristics
    tld_key = _tld_heuristic(domain, modality)
    if tld_key and tld_key in registry:
        return tld_key

    # 5. Known social media
    if (
        domain in _SOCIAL_MEDIA_DOMAINS
        or _strip_subdomain(domain) in _SOCIAL_MEDIA_DOMAINS
    ):
        return "social_media_web_text"

    # 6. Modality default
    if modality == "web_table":
        return "general_web_table"
    if modality == "pdf":
        return "academic_pdf"

    # 7. Final fallback
    return "unknown_web"


def get_source_trust(source_id: str, registry: dict[str, dict]) -> float:
    """Return ST_i for a source_id, or DEFAULT_SOURCE_TRUST if not found."""
    entry = registry.get(source_id)
    return entry["source_trust"] if entry else DEFAULT_SOURCE_TRUST


def verify_generic_fallback_coverage(registry: dict[str, dict]) -> list[str]:
    """Return missing generic fallback source_ids.

    For every (source_type, modality) pair that appears in the registry,
    there must be an entry whose source_id == f'{source_type}_{modality}'.
    Returns an empty list when coverage is complete.
    """
    combos = {(e["source_type"], e["modality"]) for e in registry.values()}
    return [
        f"{st}_{mod}"
        for st, mod in sorted(combos)
        if f"{st}_{mod}" not in registry
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_domain(domain: str) -> str:
    """Strip TLD and convert to snake_case for registry key lookup.

    reuters.com → reuters, wikipedia.org → wikipedia,
    bbc.co.uk → bbc (compound TLD stripped), indiatoday.in → indiatoday.
    """
    parts = domain.split(".")
    if len(parts) >= 3:
        compound = ".".join(parts[-2:])
        if compound in _COMPOUND_TLDS:
            # Strip both TLD components (e.g. ".co.uk") → keep brand only
            name = ".".join(parts[:-2])
            return name.replace(".", "_").replace("-", "_")
    if len(parts) >= 2:
        # Strip single TLD: reuters.com → reuters
        name = ".".join(parts[:-1])
        return name.replace(".", "_").replace("-", "_")
    return domain.replace(".", "_").replace("-", "_")


def _strip_subdomain(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) > 2:
        return ".".join(parts[1:])
    return domain


def _tld_heuristic(domain: str, modality: str) -> str | None:
    """Return a registry source_id based on domain TLD pattern.

    Maps to the actual registry keys: government_web_text, government_pdf,
    academic_pdf, general_web_text.  Returns None for .org (too heterogeneous).
    """
    gov_suffixes = (
        ".gov",
        ".gov.uk",
        ".gov.au",
        ".gov.in",
        ".gov.za",
        ".gov.ng",
        ".gov.gh",
        ".gov.ke",
        ".gouv.fr",
        ".go.ke",
        ".go.ug",
        ".go.tz",
        ".nhs.uk",
    )
    edu_suffixes = (".edu", ".ac.uk", ".ac.in", ".ac.za", ".edu.au")

    for s in gov_suffixes:
        if domain.endswith(s) or domain == s.lstrip("."):
            return "government_pdf" if modality == "pdf" else "government_web_text"
    for s in edu_suffixes:
        if domain.endswith(s) or domain == s.lstrip("."):
            return "academic_pdf" if modality == "pdf" else "general_web_text"
    return None
