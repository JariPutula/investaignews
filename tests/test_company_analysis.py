"""Source integrity, cache reuse and isolation of paid AI calls."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from learning.company_analysis import Analysis, Claim, Evidence, generate_analysis, load_analysis, make_packet, packet_fingerprint, validate_analysis
from market_data.annual_reports import excerpts_from_html
from market_data.companies import get_company
from market_data.sec import SecDataError
from tests.test_sec_companies import extract


def packet_fixture():
    report = extract("AMZN")
    report["raw_sha256"] = "raw-checksum"
    row = report["rows"][-1]
    annual = {"accession": row["sources"]["revenue"]["accession"], "cik": report["cik"],
              "sha256": "annual-checksum", "filed": "2026-08-01", "url": "https://www.sec.gov/example.htm",
              "excerpts": [{"id": "R0001", "text": "Operating cash flow increased primarily due to higher net income."}]}
    return make_packet(report, row, annual)


def analysis_fixture():
    metric = Evidence(source_id="M2025_operating_cash_flow", quote="")
    report = Evidence(source_id="R0001", quote="Operating cash flow increased")
    return Analysis(summary="Harjoitusanalyysi.", observations=[Claim(text="Rahavirta on positiivinen.", evidence=[metric])],
                    explanations=[Claim(text="Raportti selittää rahavirran kasvua.", evidence=[report])],
                    counterpoints=[Claim(text="Yksi vuosi ei ratkaise pysyvyyttä.", evidence=[metric])],
                    questions=[Claim(text="Jatkuuko rahavirta samanlaisena?", evidence=[metric])])


class AnalysisTests(unittest.TestCase):
    def test_forward_plan_is_not_an_explanation_of_completed_year(self):
        packet = packet_fixture()
        quote = "We expect to increase capital expenditures in fiscal year 2027 relative to fiscal year 2026."
        packet["sources"]["R0002"] = {"text": quote, "url": "https://www.sec.gov/fixture.htm"}
        analysis = analysis_fixture()
        analysis.explanations[0].evidence = [Evidence(source_id="R0002", quote=quote)]
        with self.assertRaisesRegex(SecDataError, "tulevaisuutta"):
            validate_analysis(analysis, packet)

    def test_derived_metric_keeps_actual_input_sources(self):
        packet = packet_fixture()
        source = packet["sources"]["M2025_free_cash_flow"]
        self.assertEqual(set(source["fact_sources"]), {"operating_cash_flow", "capex"})
        self.assertEqual(validate_analysis(analysis_fixture(), packet).summary, "Harjoitusanalyysi.")

    def test_unknown_source_and_fabricated_quote_are_rejected(self):
        for field, value in (("source_id", "R9999"), ("quote", "This invented quote is not in the filing")):
            analysis = analysis_fixture()
            setattr(analysis.explanations[0].evidence[0], field, value)
            with self.subTest(field=field), self.assertRaises(SecDataError):
                validate_analysis(analysis, packet_fixture())

    def test_explanation_needs_report_and_observation_needs_numbers(self):
        for group, evidence in (("explanations", Evidence(source_id="M2025_revenue", quote="")),
                                ("observations", Evidence(source_id="R0001", quote="Operating cash flow increased"))):
            analysis = analysis_fixture()
            getattr(analysis, group)[0].evidence = [evidence]
            with self.subTest(group=group), self.assertRaises(SecDataError):
                validate_analysis(analysis, packet_fixture())

    def test_changed_numbers_or_report_cannot_reuse_old_analysis(self):
        packet = packet_fixture()
        for key in ("annual_sha256", "raw_sha256", "as_of", "fiscal_year"):
            modified = deepcopy(packet)
            modified[key] = "different"
            self.assertNotEqual(packet_fingerprint(packet), packet_fingerprint(modified))
        modified = deepcopy(packet)
        modified["sources"]["M2025_revenue"]["text"] = "changed value"
        self.assertNotEqual(packet_fingerprint(packet), packet_fingerprint(modified))

    def test_wrong_company_filing_is_rejected(self):
        report = extract("AMZN")
        row = report["rows"][-1]
        with self.assertRaises(SecDataError):
            make_packet(report, row, {"accession": "wrong", "cik": report["cik"]})

    def test_paid_call_is_cached_and_failure_does_not_write_analysis(self):
        packet = packet_fixture()
        client = Mock()
        client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=analysis_fixture(), refusal=None))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500))
        with TemporaryDirectory() as directory, patch("learning.company_analysis.analysis_path", return_value=Path(directory)/"analysis.json"):
            first = generate_analysis(get_company("AMZN"), packet, "test-key", client=client)
            second = generate_analysis(get_company("AMZN"), packet, "", client=client)
            self.assertEqual(first, second)
            client.chat.completions.parse.assert_called_once()
            self.assertFalse(client.chat.completions.parse.call_args.kwargs["store"])
            self.assertNotIn("test-key", (Path(directory)/"analysis.json").read_text())
        client.chat.completions.parse.side_effect = RuntimeError("sensitive-key")
        with TemporaryDirectory() as directory, patch("learning.company_analysis.analysis_path", return_value=Path(directory)/"analysis.json"):
            with self.assertRaises(SecDataError) as error:
                generate_analysis(SimpleNamespace(ticker="AMZN", data_dir=Path(directory)), packet, "test-key", client=client)
            self.assertNotIn("sensitive-key", str(error.exception))
            self.assertFalse((Path(directory)/"analysis.json").exists())

    def test_hidden_report_content_is_not_sent(self):
        html = '<html><head>cash flow hidden-head</head><body><ix:header><ix:hidden>secret-instruction</ix:hidden></ix:header>'
        html += '<p>Operating cash flow and capital expenditure determine liquidity. ' * 200
        html += '<script>hidden-script</script></body></html>'
        excerpts = excerpts_from_html(html.encode())
        text = " ".join(item["text"] for item in excerpts)
        self.assertNotIn("secret-instruction", text)
        self.assertNotIn("hidden-script", text)
        self.assertNotIn("hidden-head", text)


if __name__ == "__main__":
    unittest.main()
