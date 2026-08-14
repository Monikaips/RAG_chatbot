import os
import json
import re

from dotenv import load_dotenv
from rag.openai_usage import get_tracked_chat_openai
from langchain_core.prompts import ChatPromptTemplate

from rag.report_columns import (
    DEFAULT_RANKING_COLUMNS,
    build_column_instructions,
    build_column_schema,
    column_to_field,
)


load_dotenv()


def _clean_json_content(content: str) -> str:
    """Strip markdown fences if the model wraps JSON."""

    text = (content or "").strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def get_matcher(output_columns=None):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    llm = get_tracked_chat_openai(model="gpt-4o-mini")

    columns = output_columns or DEFAULT_RANKING_COLUMNS

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an experienced technical recruiter and ATS resume screening assistant.

Your task is to evaluate how well a candidate's resume matches a specific Job Description (JD),
and to fill the requested report columns.

IMPORTANT RULES

1. Use ONLY the provided Job Description and Candidate Resume.
2. Never invent candidate information.
3. Never assume a skill, tool, certification, responsibility, or experience that is not explicitly stated.
4. Score candidates strictly against the provided JD.
5. Re-evaluate every new JD independently.
6. If information is missing or unclear, mark it as "Not Mentioned" and score conservatively.
7. Do not inflate scores.
8. Be consistent and objective.
9. Return ONLY valid JSON.
10. Do not return markdown.
11. Do not include ```json.

SCORING RUBRIC

Use the following default weighting:

- Core skills, tools, technologies explicitly required in the JD: 40%
- Relevant experience compared to JD requirements: 20%
- Seniority and scope alignment with JD responsibilities: 20%
- Domain or industry relevance (if specified): 10%
- Education and certifications (if specified): 10%

If any category is not mentioned in the JD, redistribute its weight proportionally across the remaining categories.

EVALUATION PROCESS

For each candidate:

1. Identify core JD requirements.
2. Compare resume evidence against those requirements.
3. Identify explicitly matched skills.
4. Identify important missing skills or requirements.
5. Assess experience level alignment.
6. Assess seniority/responsibility alignment.
7. Assess domain relevance if applicable.
8. Assess education/certification relevance if applicable.
9. Produce a final score from 0-100.
10. Fill every requested report column listed below.

VERDICT RULES

- 85-100 = Strong Fit
- 65-84 = Possible Fit
- 0-64 = Weak Fit

REQUESTED REPORT COLUMNS

{column_guide}

Return EXACTLY this JSON structure (include every key shown):

{schema_json}

Always include:
- "score" as an integer 0-100
- "matched_skills" as a list
- "missing_skills" as a list
- "reason" as a short explanation

For Key Strengths / Key Gaps style list fields, return at most 3 short bullet strings.

JOB DESCRIPTION:

{jd_text}

CANDIDATE RESUME:

{candidate_text}
"""
        )
    ])

    return prompt | llm, columns


def evaluate_candidate(
    jd_text,
    candidate_text,
    output_columns=None,
):

    matcher, columns = get_matcher(output_columns)

    schema = build_column_schema(columns)
    column_guide = build_column_instructions(columns)
    schema_json = json.dumps(schema, indent=4)

    response = matcher.invoke({
        "jd_text": jd_text,
        "candidate_text": candidate_text,
        "column_guide": column_guide,
        "schema_json": schema_json,
    })

    fallback = {
        "score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "key_strengths": [],
        "key_gaps": [],
        "verdict": "Weak Fit",
        "reason": "Candidate evaluation failed.",
    }

    try:

        result = json.loads(
            _clean_json_content(response.content)
        )

        if not isinstance(result, dict):
            return fallback

        evaluation = {
            "score": result.get("score", 0),
            "matched_skills": result.get(
                "matched_skills",
                []
            ),
            "missing_skills": result.get(
                "missing_skills",
                []
            ),
            "reason": result.get(
                "reason",
                "No explanation available."
            ),
        }

        try:
            evaluation["score"] = int(evaluation["score"])
        except (TypeError, ValueError):
            evaluation["score"] = 0

        # Pull every requested column field from the model response
        for column in columns:
            field = column_to_field(column)

            if field in evaluation:
                continue

            if field in result:
                evaluation[field] = result[field]
            elif field in (
                "key_strengths",
                "key_gaps",
                "matched_skills",
                "missing_skills",
            ):
                evaluation[field] = []
            elif field == "verdict":
                evaluation[field] = _verdict_from_score(
                    evaluation["score"]
                )
            else:
                evaluation[field] = "Not Mentioned"

        if "verdict" not in evaluation:
            evaluation["verdict"] = _verdict_from_score(
                evaluation["score"]
            )

        if "key_strengths" not in evaluation:
            evaluation["key_strengths"] = result.get(
                "key_strengths",
                []
            )

        if "key_gaps" not in evaluation:
            evaluation["key_gaps"] = result.get(
                "key_gaps",
                []
            )

        return evaluation

    except json.JSONDecodeError:

        return fallback


def _verdict_from_score(score: int) -> str:

    try:
        value = int(score)
    except (TypeError, ValueError):
        value = 0

    if value >= 85:
        return "Strong Fit"

    if value >= 65:
        return "Possible Fit"

    return "Weak Fit"
