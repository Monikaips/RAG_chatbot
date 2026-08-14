"""
Capture actual OpenAI usage from each API response and persist it.

Flow:
    OpenAI / LangChain call
        -> actual response
        -> extract response.usage / usage_metadata
        -> calculate cost from model pricing
        -> save usage record
        -> return original response to existing logic
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatResult, LLMResult
from langchain_openai import ChatOpenAI

from rag.openai_pricing import calculate_cost, normalize_model_name
from rag.usage_store import insert_usage_record


logger = logging.getLogger(__name__)

_operation: ContextVar[str | None] = ContextVar("usage_operation", default=None)
_resume_id: ContextVar[str | None] = ContextVar("usage_resume_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("usage_job_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("usage_user_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("usage_run_id", default=None)
_db_path: ContextVar[str | None] = ContextVar("usage_db_path", default=None)


@contextmanager
def usage_context(
    operation: str | None = None,
    resume_id: str | None = None,
    job_id: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
    db_path: str | None = None,
):
    tokens = []

    if operation is not None:
        tokens.append((_operation, _operation.set(operation)))
    if resume_id is not None:
        tokens.append((_resume_id, _resume_id.set(resume_id)))
    if job_id is not None:
        tokens.append((_job_id, _job_id.set(job_id)))
    if user_id is not None:
        tokens.append((_user_id, _user_id.set(user_id)))
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    if db_path is not None:
        tokens.append((_db_path, _db_path.set(db_path)))

    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_usage_context() -> dict:
    return {
        "operation": _operation.get(),
        "resume_id": _resume_id.get(),
        "job_id": _job_id.get(),
        "user_id": _user_id.get() or os.getenv("ATS_USER_ID", "local"),
        "run_id": _run_id.get(),
        "db_path": _db_path.get(),
    }


def _to_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage_from_mapping(usage) -> dict:
    """
    Read actual token counts from an OpenAI usage object/dict.

    Does not estimate. Missing fields stay None.
    """

    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "usage_present": False,
        }

    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    elif not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

    input_tokens = _to_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    output_tokens = _to_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _to_int(usage.get("total_tokens"))

    usage_present = any(
        value is not None
        for value in (input_tokens, output_tokens, total_tokens)
    )

    if (
        total_tokens is None
        and input_tokens is not None
        and output_tokens is not None
    ):
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_present": usage_present,
    }


def extract_usage_from_llm_result(response: LLMResult) -> dict:
    llm_output = response.llm_output or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    request_id = llm_output.get("id")

    usage = extract_usage_from_mapping(llm_output.get("token_usage"))

    if not usage["usage_present"] and response.generations:
        first = response.generations[0]
        if first:
            message = getattr(first[0], "message", None)
            metadata = getattr(message, "usage_metadata", None)
            usage = extract_usage_from_mapping(metadata)
            if not model and message is not None:
                response_metadata = getattr(message, "response_metadata", {}) or {}
                model = response_metadata.get("model_name") or response_metadata.get("model")
            if not request_id and message is not None:
                response_metadata = getattr(message, "response_metadata", {}) or {}
                request_id = response_metadata.get("id")

    return {
        "model": normalize_model_name(model),
        "request_id": request_id,
        **usage,
    }


def record_actual_usage(
    *,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    usage_present: bool,
    request_id: str | None = None,
    operation: str | None = None,
    resume_id: str | None = None,
    job_id: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
    db_path: str | None = None,
) -> str | None:
    context = current_usage_context()

    if not usage_present:
        logger.warning(
            "OpenAI response had no usage object. "
            "operation=%s resume_id=%s job_id=%s model=%s request_id=%s. "
            "Token counts will be stored as null; nothing was estimated.",
            operation or context["operation"],
            resume_id or context["resume_id"],
            job_id or context["job_id"],
            model,
            request_id,
        )
        costs = {
            "input_cost": None,
            "output_cost": None,
            "total_cost": None,
        }
    else:
        costs = calculate_cost(model, input_tokens, output_tokens)

    record = {
        "user_id": user_id or context["user_id"],
        "job_id": job_id or context["job_id"],
        "resume_id": resume_id or context["resume_id"],
        "run_id": run_id or context["run_id"],
        "operation": operation or context["operation"],
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost": costs["input_cost"],
        "output_cost": costs["output_cost"],
        "total_cost": costs["total_cost"],
        "request_id": request_id,
    }

    row_id = insert_usage_record(record, db_path or context["db_path"])
    logger.info(
        "Recorded OpenAI usage operation=%s model=%s input=%s output=%s cost=%s request_id=%s",
        record["operation"],
        record["model"],
        record["input_tokens"],
        record["output_tokens"],
        record["total_cost"],
        request_id,
    )
    return row_id


def record_from_llm_result(response: LLMResult) -> str | None:
    extracted = extract_usage_from_llm_result(response)
    return record_actual_usage(
        model=extracted.get("model"),
        input_tokens=extracted.get("input_tokens"),
        output_tokens=extracted.get("output_tokens"),
        total_tokens=extracted.get("total_tokens"),
        usage_present=extracted.get("usage_present", False),
        request_id=extracted.get("request_id"),
    )


def record_from_chat_result(result: ChatResult) -> str | None:
    """Record usage from ChatOpenAI._generate ChatResult."""

    llm_output = result.llm_output or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    request_id = llm_output.get("id")
    usage = extract_usage_from_mapping(llm_output.get("token_usage"))

    if not usage["usage_present"] and result.generations:
        message = getattr(result.generations[0], "message", None)
        usage = extract_usage_from_mapping(
            getattr(message, "usage_metadata", None)
        )
        if message is not None:
            response_metadata = getattr(message, "response_metadata", {}) or {}
            model = model or response_metadata.get("model_name") or response_metadata.get("model")
            request_id = request_id or response_metadata.get("id") or getattr(message, "id", None)

    return record_actual_usage(
        model=normalize_model_name(model),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        usage_present=usage["usage_present"],
        request_id=request_id,
    )


def record_from_openai_response(response, model: str | None = None) -> str | None:
    """Record usage from a raw OpenAI SDK response (chat or embeddings)."""

    usage = getattr(response, "usage", None)
    extracted = extract_usage_from_mapping(usage)
    response_model = (
        getattr(response, "model", None)
        or model
    )
    request_id = getattr(response, "id", None)

    return record_actual_usage(
        model=normalize_model_name(response_model),
        input_tokens=extracted["input_tokens"],
        output_tokens=extracted["output_tokens"],
        total_tokens=extracted["total_tokens"],
        usage_present=extracted["usage_present"],
        request_id=request_id,
    )


class UsageTrackingCallback(BaseCallbackHandler):
    """Records one usage row per successful LLM API completion."""

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            record_from_llm_result(response)
        except Exception:
            logger.exception(
                "Failed to record OpenAI usage from LLM response."
            )


class TrackedChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI that records actual token usage from each completion.

    Recording happens on the ChatResult returned by the real API call,
    not from estimated prompt length.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        try:
            record_from_chat_result(result)
        except Exception:
            logger.exception(
                "Failed to record OpenAI usage from chat completion."
            )
        return result


def get_tracked_chat_openai(model: str = "gpt-4o-mini") -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")

    return TrackedChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
    )
