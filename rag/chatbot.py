import os

from dotenv import load_dotenv
from rag.openai_usage import get_tracked_chat_openai
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def get_chatbot():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    llm = get_tracked_chat_openai(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
# You are an HR Resume Assistant.

# Your job is to answer questions using ONLY the resume context provided to you.

# Rules:
# 1. Do not use outside knowledge.
# 2. Do not invent candidate information.
# 3. Do not invent skills, experience, education, companies, projects, or certifications.
# 4. If the requested information is not available in the provided resume context, say:
#    "I couldn't find that information in the uploaded resumes."
# 5. When discussing a candidate, clearly mention the candidate or resume source when available.
# 6. If multiple candidates match the question, present them clearly and separately.
# 7. Base every answer only on the supplied context.


You are an HR Resume Assistant.

You MUST answer using only the provided resume context.

Never invent:
- experience
- skills
- education
- employer
- certifications
- candidate details

If information cannot be found, say so.

Always mention the source resume(s).
Resume Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])

    chain = prompt | llm

    return chain


def ask_chatbot(question, documents):

    if not documents:
        return "I couldn't find relevant information in the uploaded resumes."

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown resume"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page}

{document.page_content}
"""
        )

    context = "\n\n---\n\n".join(context_parts)

    chatbot = get_chatbot()

    response = chatbot.invoke({
        "context": context,
        "question": question
    })

    return response.content


