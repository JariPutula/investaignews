"""The user sees the actual failure category; credentials never enter diagnostics."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import APITimeoutError, AuthenticationError, LengthFinishReasonError, RateLimitError
from openai.types.chat import ChatCompletion

from learning.analysis_errors import explain_api_error, record_api_error
from learning.analysis_index import list_saved_analyses
from learning.company_analysis import MODEL, PROMPT_VERSION
from tests.test_company_analysis import analysis_fixture


class AnalysisErrorTests(unittest.TestCase):
    def test_timeout_quota_and_truncation_have_distinct_actionable_messages(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        timeout = APITimeoutError(request=request)
        quota = RateLimitError("sensitive body", response=httpx.Response(429, request=request),
                               body={"code": "insufficient_quota"})
        completion = ChatCompletion(id="fixture", created=0, model=MODEL, object="chat.completion",
                                    choices=[{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": ""}}])
        length = LengthFinishReasonError(completion=completion)
        self.assertIn("määräajassa", explain_api_error(timeout))
        self.assertIn("laskutusraja", explain_api_error(quota))
        self.assertIn("pituusrajaan", explain_api_error(length))
        self.assertNotIn("API-yhteys", explain_api_error(length))

    def test_diagnostic_keeps_safe_codes_but_no_key_or_response_body(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        error = AuthenticationError("API key sk-private-test was rejected", response=httpx.Response(401, request=request),
                                    body={"code": "invalid_api_key", "message": "sk-private-test", "prompt": "private payload"})
        with TemporaryDirectory() as directory:
            company = SimpleNamespace(ticker="NVDA", data_dir=Path(directory))
            result = record_api_error(company, {"fiscal_year": 2026}, error, model=MODEL, output_limit=4000)
            written = next((Path(directory)/"analyses"/"errors").glob("*.json")).read_text()
            self.assertIn("API-avaimen", str(result))
            self.assertEqual(result.details["error_code"], "invalid_api_key")
            self.assertEqual(result.details["http_status"], 401)
            self.assertNotIn("sk-private-test", written + str(result))
            self.assertNotIn("private payload", written)

    def test_saved_index_ignores_invalid_and_obsolete_results(self):
        with TemporaryDirectory() as directory:
            company = SimpleNamespace(ticker="AMZN", name="Amazon", data_dir=Path(directory))
            target = Path(directory)/"analyses"
            target.mkdir()
            saved = {"model": MODEL, "prompt_version": PROMPT_VERSION, "fiscal_year": 2025,
                     "fingerprint": "a" * 64, "created_at": "2026-11-17T10:00:00+00:00",
                     "analysis": analysis_fixture().model_dump()}
            (target/('2025_' + 'a'*20 + '.json')).write_text(json.dumps(saved))
            (target/"2026_invalid.json").write_text("{invalid")
            (target/"2026_obsolete.json").write_text(json.dumps({**saved, "model": "old-model"}))
            with patch("learning.analysis_index.COMPANIES", {"AMZN": company}):
                found = list_saved_analyses()
            self.assertEqual([(item["ticker"], item["fiscal_year"]) for item in found], [("AMZN", 2025)])

    def test_saved_index_prefers_company_brief_and_keeps_legacy_file(self):
        from learning.company_brief import BRIEF_VERSION
        from tests.test_company_brief import brief_fixture
        with TemporaryDirectory() as directory:
            company = SimpleNamespace(ticker="AMZN", name="Amazon", data_dir=Path(directory))
            target = Path(directory)/"analyses"
            target.mkdir()
            cash = {"model": MODEL, "prompt_version": PROMPT_VERSION, "fiscal_year": 2025, "fingerprint": "a"*64,
                    "created_at": "2026-11-18T12:00:00+00:00", "analysis": analysis_fixture().model_dump()}
            brief = {"model": MODEL, "brief_version": BRIEF_VERSION, "analysis_scope": "company_brief", "fiscal_year": 2025,
                     "fingerprint": "b"*64, "created_at": "2026-11-18T11:00:00+00:00", "analysis": brief_fixture().model_dump()}
            old_path = target/('2025_' + 'a'*20 + '.json')
            old_path.write_text(json.dumps(cash))
            (target/('brief_2025_' + 'b'*20 + '.json')).write_text(json.dumps(brief))
            with patch("learning.analysis_index.COMPANIES", {"AMZN": company}):
                found = list_saved_analyses()
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["analysis_scope"], "company_brief")
            self.assertTrue(old_path.exists())
