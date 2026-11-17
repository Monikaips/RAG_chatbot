import os
import json

from dotenv import load_dotenv
from rag.openai_usage import get_tracked_chat_openai
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def get_extractor():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    llm = get_tracked_chat_openai(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an HR resume information extraction assistant.

Your job is to extract candidate information ONLY from the
resume text provided.

Extract these fields:

- Candidate Name
- Position
- Skills
- Email ID
- Mobile Number
- Location
- Visa Category
- Experience
- Last Employer
- Last Employer Period

IMPORTANT RULES:

1. Use ONLY information explicitly available in the resume.
2. Never invent or guess candidate information.
3. If information is unavailable, return "Not Mentioned".
4. Do not infer Visa Category from nationality, location,
   education, employer, or any other information.
5. Skills must be returned as a list.
6. Experience should represent total professional experience
   if it is explicitly stated or clearly available in the resume.
7. Email and phone number must come directly from the resume.
8. Last Employer and Last Employer Period MUST be taken
   from the single most recent work experience entry,
   decided by employment dates — NOT by which job looks
   biggest, NOT by client/deputation name, and NOT by
   reading order if dates disagree.
9. How to choose the most recent employer:
   - Scan EVERY work-experience / employment block.
   - Compare start and end dates for each block.
   - Prefer the block whose end date is latest.
   - "Present", "Till date", "Current", or ongoing means
     that block is more recent than any ended job.
   - If the same company appears more than once
     (e.g. "Rejoined"), use the LATEST stint only.
   - Example: "Jun 2025 – May 2026" at Sports Mechanics
     beats "Dec 2020 – Nov 2023" at Sonata — return
     Sports Mechanics, not Sonata.
10. last_employer = company / organization name ONLY.
    NEVER put the job title in last_employer.
    - Correct: "Sports Mechanics India Pvt Ltd (SMIPL)"
    - Correct freelance: "Independent" or
      "Freelance / Independent"
    - Wrong: "Freelance Network Security Engineer"
      (that is the job title, not the employer)
    - Wrong: "Senior Security Engineer"
    For lines like
    "Job Title | Company Name", use the part AFTER "|"
    as last_employer, never the title before "|".
    Do NOT use a deputed client as the employer unless
    no company name is given.
11. last_employer_period = the date range for THAT same
    latest stint only (e.g. "Feb 2025 – Present",
    "Jun 2025 – May 2026"). Keep the resume's wording
    when clear.
    
12. Return ONLY valid JSON.
13. Do not include markdown or ```json formatting.

Return exactly this structure:

{{
    "candidate_name": "value",
    "position": "value",
    "skills": ["skill1", "skill2"],
    "email": "value",
    "mobile": "value",
    "location": "value",
    "visa_category": "value",
    "experience": "value",
    "last_employer": "value",
    "last_employer_period": "value"
}}

Resume:

{resume_text}
"""
        )
    ])

    return prompt | llm


def extract_candidate_details(resume_text):

    extractor = get_extractor()

    response = extractor.invoke({
        "resume_text": resume_text
    })

    try:

        candidate = json.loads(response.content)

        return candidate

    except json.JSONDecodeError:

        return {
            "candidate_name": "Not Mentioned",
            "position": "Not Mentioned",
            "skills": [],
            "email": "Not Mentioned",
            "mobile": "Not Mentioned",
            "location": "Not Mentioned",
            "visa_category": "Not Mentioned",
            "experience": "Not Mentioned",
            "last_employer": "Not Mentioned",
            "last_employer_period": "Not Mentioned",
        }


def group_resumes(documents):

    grouped = {}

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown Resume"
        )

        if source not in grouped:
            grouped[source] = []

        grouped[source].append(
            document.page_content
        )

    combined_resumes = {}

    for source, pages in grouped.items():

        combined_resumes[source] = "\n\n".join(pages)

    return combined_resumes