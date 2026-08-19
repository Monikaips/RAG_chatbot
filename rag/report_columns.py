"""
Configurable ranking report columns.

Users pick columns in the Streamlit UI. Known columns map to
extracted/evaluated fields; custom columns are filled by the LLM.
"""

DEFAULT_RANKING_COLUMNS = [
    "Candidate Name",
    "Match Score (/100)",
    "Key Strengths",
    "Key Gaps",
    "Verdict",
    "Current Location/Area",
    "Years of Exp",
    "Email",
    "Mobile",
]

PRESET_RANKING_COLUMNS = [
    "Candidate Name",
    "Match Score (/100)",
    "Key Strengths",
    "Key Gaps",
    "Verdict",
    "Current Location/Area",
    "Years of Exp",
    "Email",
    "Mobile",
    "Matched Skills",
    "Missing Skills",
    "Visa Category",
    "Position",
    "Resume File",
    "Reason",
]

# Internal field key used in candidate/evaluation dicts
COLUMN_FIELD_MAP = {
    "Candidate Name": "candidate_name",
    "Match Score (/100)": "score",
    "Key Strengths": "key_strengths",
    "Key Gaps": "key_gaps",
    "Verdict": "verdict",
    "Current Location/Area": "location",
    "Years of Exp": "experience",
    "Email": "email",
    "Mobile": "mobile",
    "Matched Skills": "matched_skills",
    "Missing Skills": "missing_skills",
    "Visa Category": "visa_category",
    "Position": "position",
    "Resume File": "source",
    "Reason": "reason",
}

# Columns the JD matcher LLM should compute (not just copy from extractor)
EVALUATION_COLUMNS = {
    "Match Score (/100)",
    "Key Strengths",
    "Key Gaps",
    "Verdict",
    "Matched Skills",
    "Missing Skills",
    "Reason",
}

# Guidance injected into the matcher prompt per column
COLUMN_INSTRUCTIONS = {
    "Candidate Name": "Full name from the resume.",
    "Match Score (/100)": "Integer score from 0 to 100.",
    "Key Strengths": (
        "List of 2-3 short bullet strings highlighting "
        "strongest JD-aligned strengths. Max 3 items."
    ),
    "Key Gaps": (
        "List of 2-3 short bullet strings highlighting "
        "important gaps vs the JD. Max 3 items."
    ),
    "Verdict": (
        'Exactly one of: "Strong Fit", "Possible Fit", "Weak Fit". '
        "Use 85-100 Strong Fit, 65-84 Possible Fit, 0-64 Weak Fit."
    ),
    "Current Location/Area": (
        "Current location/area from the resume, or Not Mentioned."
    ),
    "Years of Exp": (
        "Total years of professional experience if stated, "
        "else Not Mentioned."
    ),
    "Email": "Email address from the resume, or Not Mentioned.",
    "Mobile": "Mobile/phone from the resume, or Not Mentioned.",
    "Matched Skills": "List of skills present that match the JD.",
    "Missing Skills": (
        "List of important JD skills missing from the resume."
    ),
    "Visa Category": "Visa/work authorization if stated, else Not Mentioned.",
    "Position": "Current or target role/title from the resume.",
    "Resume File": "Leave as empty string; filled by the system.",
    "Reason": "Short explanation for the score and verdict.",
}


def column_to_field(column_name: str) -> str:
    """Map a display column name to a stable internal field key."""

    if column_name in COLUMN_FIELD_MAP:
        return COLUMN_FIELD_MAP[column_name]

    slug = (
        column_name.strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug.strip("_") or "custom_field"


def parse_custom_columns(raw_text: str) -> list[str]:
    """Split comma/newline separated custom column names."""

    if not raw_text or not str(raw_text).strip():
        return []

    parts = []

    for chunk in str(raw_text).replace("\n", ",").split(","):
        name = chunk.strip()
        if name:
            parts.append(name)

    return parts


def merge_output_columns(
    selected_presets: list[str],
    custom_text: str = "",
) -> list[str]:
    """Combine multiselect presets with typed custom columns."""

    columns = []

    for column in selected_presets or []:
        if column and column not in columns:
            columns.append(column)

    for column in parse_custom_columns(custom_text):
        if column not in columns:
            columns.append(column)

    return columns


def format_cell_value(value) -> str:
    """Normalize list/bullet values for table display."""

    if value is None:
        return "Not Mentioned"

    if isinstance(value, list):
        cleaned = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

        if not cleaned:
            return "Not Mentioned"

        if len(cleaned) == 1:
            return cleaned[0]

        return " • ".join(cleaned)

    text = str(value).strip()

    return text if text else "Not Mentioned"


def build_ranking_row(
    candidate: dict,
    columns: list[str],
    rank: int | None = None,
) -> dict:
    """Build one ranking table row from candidate + selected columns."""

    row = {}

    if rank is not None:
        row["Rank"] = rank

    for column in columns:
        field = column_to_field(column)
        value = candidate.get(field, "Not Mentioned")

        if field == "score":
            try:
                row[column] = int(value)
            except (TypeError, ValueError):
                row[column] = 0
        else:
            row[column] = format_cell_value(value)

    return row


def build_column_schema(columns: list[str]) -> dict:
    """
    JSON-shaped schema description for the matcher prompt.
    Always includes score for sorting.
    """

    schema = {
        "score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "reason": "short explanation",
    }

    for column in columns:
        field = column_to_field(column)

        if field in ("score", "matched_skills", "missing_skills", "reason"):
            continue

        if field in ("key_strengths", "key_gaps"):
            schema[field] = ["bullet 1", "bullet 2"]
        elif field == "verdict":
            schema[field] = "Strong Fit | Possible Fit | Weak Fit"
        else:
            schema[field] = "value"

    return schema


def build_column_instructions(columns: list[str]) -> str:
    """Human-readable per-column guidance for the matcher prompt."""

    lines = []

    for column in columns:
        field = column_to_field(column)
        guidance = COLUMN_INSTRUCTIONS.get(
            column,
            (
                f'Extract or infer "{column}" from the resume vs JD. '
                "If unavailable, return Not Mentioned. "
                "Do not invent facts."
            ),
        )
        lines.append(f'- "{column}" (JSON key: "{field}"): {guidance}')

    return "\n".join(lines)
