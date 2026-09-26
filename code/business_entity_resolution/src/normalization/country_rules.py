"""
Country-aware token sets used only for AUXILIARY views (e.g. name content
tokens). Main views never depend on country.

Every entry is justified by Stage 1 observations (experiments/stage1/
observed_patterns.md); France entries come from inspecting unlabeled test
records. Country is an open set: an unknown country gets the union of all
known rules rather than no rules.
"""

LEGAL_FORMS = {
    "us": {
        "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
        "co", "company", "plc", "llp", "lp", "pllc", "pc",
    },
    "india": {
        "pvt", "private", "ltd", "limited", "llp", "opc",
        "co", "company", "corp", "corporation", "inc",
    },
    "france": {
        "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "scop", "selarl",
        "ltd", "inc", "llc",
    },
}

# "&" is punctuation and disappears from views, so the spelled-out
# connector must go too ("Keys & Co" vs "KEYS and CO").
CONNECTORS = {
    "us": {"and"},
    "india": {"and"},
    "france": {"et"},
}

# Domain-style names ("fortunefinance.com") are ~4% of S2/S3 names.
WEB_TOKENS = {"www", "com", "net", "org"}

# Address comma-component placeholders ("NULL", "N/A") seen in ~2.7% of
# S2/S3 addresses. Compared after casefolding. Not applied to names.
PLACEHOLDERS = {"null", "n/a", "na", "n.a.", "nan", "none", "nil", "unknown", "-", "--", "."}


def country_key(country) -> str:
    if country is None:
        return ""
    return str(country).strip().casefold()


def _union(table: dict) -> set:
    return set().union(*table.values())


def get_country_rule(country) -> dict:
    key = country_key(country)
    known = key in LEGAL_FORMS

    legal = LEGAL_FORMS[key] if known else _union(LEGAL_FORMS)
    connectors = CONNECTORS[key] if known else _union(CONNECTORS)

    return {
        "country_key": key,
        "known_country": known,
        "legal_forms": frozenset(legal),
        "name_noise_tokens": frozenset(legal | connectors | WEB_TOKENS),
    }
