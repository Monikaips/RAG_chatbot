import inspect
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from rag.candidate_matcher import evaluate_candidate
from rag.openai_pricing import calculate_cost, get_model_pricing
from rag.openai_usage import (
    extract_usage_from_mapping,
    record_from_chat_result,
    record_from_llm_result,
    usage_context,
)
from rag.usage_store import (
    insert_usage_record,
    query_usage_records,
    summarize_by_resume,
    summarize_records,
)


def _llm_result(
    prompt_tokens,
    completion_tokens,
    total_tokens,
    model="gpt-4o-mini",
    request_id="req-1",
):
    usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    message = AIMessage(content="ok")
    generation = ChatGeneration(message=message)
    return LLMResult(
        generations=[[generation]],
        llm_output={
            "token_usage": usage,
            "model_name": model,
            "id": request_id,
        },
    )


class OpenAIUsageTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "usage.sqlite")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_a_real_usage_creates_record(self):
        result = _llm_result(100, 20, 120, request_id="a-1")

        with usage_context(
            operation="resume_matching",
            resume_id="resume-a.pdf",
            job_id="jd.pdf",
            db_path=self.db_path,
        ):
            row_id = record_from_llm_result(result)

        self.assertIsNotNone(row_id)
        records = query_usage_records(db_path=self.db_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["input_tokens"], 100)
        self.assertEqual(records[0]["output_tokens"], 20)
        self.assertEqual(records[0]["total_tokens"], 120)

    def test_b_input_and_output_tokens_recorded(self):
        extracted = extract_usage_from_mapping({
            "prompt_tokens": 8421,
            "completion_tokens": 1132,
            "total_tokens": 9553,
        })
        self.assertEqual(extracted["input_tokens"], 8421)
        self.assertEqual(extracted["output_tokens"], 1132)
        self.assertEqual(extracted["total_tokens"], 9553)
        self.assertTrue(extracted["usage_present"])

    def test_c_cost_uses_model_pricing(self):
        pricing = get_model_pricing("gpt-4o-mini")
        cost = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        self.assertAlmostEqual(
            cost["input_cost"],
            pricing["input_per_million"],
        )
        self.assertAlmostEqual(
            cost["output_cost"],
            pricing["output_per_million"],
        )
        self.assertAlmostEqual(
            cost["total_cost"],
            pricing["input_per_million"] + pricing["output_per_million"],
        )

        embedding_cost = calculate_cost(
            "text-embedding-3-small",
            1_000_000,
            0,
        )
        self.assertAlmostEqual(embedding_cost["input_cost"], 0.020)
        self.assertAlmostEqual(embedding_cost["output_cost"], 0.0)

    def test_d_multiple_calls_for_one_resume_are_summed(self):
        with usage_context(
            resume_id="resume-a.pdf",
            job_id="jd.pdf",
            db_path=self.db_path,
        ):
            record_from_llm_result(
                _llm_result(100, 10, 110, request_id="d-1")
            )
            record_from_llm_result(
                _llm_result(200, 20, 220, request_id="d-2")
            )

        records = query_usage_records(
            resume_id="resume-a.pdf",
            db_path=self.db_path,
        )
        by_resume = summarize_by_resume(records)
        bucket = by_resume["resume-a.pdf"]
        self.assertEqual(bucket["calls"], 2)
        self.assertEqual(bucket["input_tokens"], 300)
        self.assertEqual(bucket["output_tokens"], 30)
        self.assertEqual(bucket["total_tokens"], 330)

        expected = calculate_cost("gpt-4o-mini", 300, 30)["total_cost"]
        self.assertAlmostEqual(bucket["total_cost"], expected)

    def test_e_multiple_resumes_are_summed(self):
        with usage_context(job_id="jd.pdf", db_path=self.db_path):
            with usage_context(resume_id="a.pdf"):
                record_from_llm_result(
                    _llm_result(50, 5, 55, request_id="e-1")
                )
            with usage_context(resume_id="b.pdf"):
                record_from_llm_result(
                    _llm_result(70, 7, 77, request_id="e-2")
                )

        summary = summarize_records(
            query_usage_records(job_id="jd.pdf", db_path=self.db_path)
        )
        self.assertEqual(summary["total_resumes"], 2)
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["input_tokens"], 120)
        self.assertEqual(summary["output_tokens"], 12)
        self.assertEqual(summary["total_tokens"], 132)

    def test_f_job_level_totals_from_actual_records(self):
        insert_usage_record(
            {
                "job_id": "backend.pdf",
                "resume_id": "one.pdf",
                "operation": "resume_extraction",
                "model": "gpt-4o-mini",
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "input_cost": 0.00015,
                "output_cost": 0.00006,
                "total_cost": 0.00021,
                "request_id": "f-1",
            },
            db_path=self.db_path,
        )
        insert_usage_record(
            {
                "job_id": "backend.pdf",
                "resume_id": "two.pdf",
                "operation": "resume_matching",
                "model": "gpt-4o-mini",
                "input_tokens": 2000,
                "output_tokens": 200,
                "total_tokens": 2200,
                "input_cost": 0.00030,
                "output_cost": 0.00012,
                "total_cost": 0.00042,
                "request_id": "f-2",
            },
            db_path=self.db_path,
        )

        summary = summarize_records(
            query_usage_records(job_id="backend.pdf", db_path=self.db_path)
        )
        self.assertEqual(summary["total_resumes"], 2)
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["input_tokens"], 3000)
        self.assertEqual(summary["output_tokens"], 300)
        self.assertEqual(summary["total_tokens"], 3300)
        self.assertAlmostEqual(summary["total_cost"], 0.00063)

    def test_g_missing_usage_does_not_invent_tokens(self):
        extracted = extract_usage_from_mapping(None)
        self.assertIsNone(extracted["input_tokens"])
        self.assertIsNone(extracted["output_tokens"])
        self.assertIsNone(extracted["total_tokens"])
        self.assertFalse(extracted["usage_present"])

        cost = calculate_cost("gpt-4o-mini", None, None)
        self.assertIsNone(cost["input_cost"])
        self.assertIsNone(cost["output_cost"])
        self.assertIsNone(cost["total_cost"])

        with usage_context(db_path=self.db_path, resume_id="x.pdf"):
            record_from_llm_result(
                _llm_result(None, None, None, request_id="g-1")
            )

        records = query_usage_records(db_path=self.db_path)
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["input_tokens"])
        self.assertIsNone(records[0]["output_tokens"])
        self.assertIsNone(records[0]["total_cost"])

    def test_h_same_request_id_is_not_recorded_twice(self):
        result = _llm_result(10, 2, 12, request_id="same-id")

        with usage_context(db_path=self.db_path, resume_id="r.pdf"):
            first = record_from_llm_result(result)
            second = record_from_llm_result(result)

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        records = query_usage_records(db_path=self.db_path)
        self.assertEqual(len(records), 1)

    def test_i_evaluate_candidate_signature_unchanged(self):
        signature = inspect.signature(evaluate_candidate)
        params = list(signature.parameters)
        self.assertEqual(params[0], "jd_text")
        self.assertEqual(params[1], "candidate_text")
        self.assertIn("output_columns", params)

    def test_j_chat_result_usage_is_recorded_and_costed(self):
        message = AIMessage(content="ok")
        result = ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                },
                "model_name": "gpt-4o-mini",
                "id": "chatcmpl-calc-1",
            },
        )

        with usage_context(
            operation="resume_matching",
            resume_id="resume-a.pdf",
            db_path=self.db_path,
        ):
            row_id = record_from_chat_result(result)

        self.assertIsNotNone(row_id)
        records = query_usage_records(db_path=self.db_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["input_tokens"], 1000)
        self.assertEqual(records[0]["output_tokens"], 200)

        expected = calculate_cost("gpt-4o-mini", 1000, 200)
        self.assertAlmostEqual(
            records[0]["input_cost"],
            expected["input_cost"],
        )
        self.assertAlmostEqual(
            records[0]["output_cost"],
            expected["output_cost"],
        )
        self.assertAlmostEqual(
            records[0]["total_cost"],
            1000 * (0.150 / 1_000_000) + 200 * (0.600 / 1_000_000),
        )


if __name__ == "__main__":
    unittest.main()
