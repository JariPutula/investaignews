"""Financial assumptions, dated price arithmetic and beginner-facing examples."""

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.company_brief import InvestorInterpretation, make_packet, validate_brief
from learning.metric_lessons import metric_lessons
from market_data.scenarios import default_assumptions, project_scenarios, save_assumptions, load_assumptions
from market_data.sec import SecDataError
from market_data.valuation import validate_quote, valuation_context, fetch_valuation
from tests.test_company_brief import brief_packet, brief_fixture
from tests.test_sec_companies import extract
from market_data.companies import get_company


def quote_fixture():
    return {"ticker": "AMZN", "currency": "USD", "price": 10., "shares": 200., "market_cap": 2000.,
            "quoted_at": "2026-11-18T21:00:00+00:00", "retrieved_at": "2026-11-19T12:00:00+00:00",
            "source_url": "https://finance.yahoo.com/quote/AMZN/", "shares_date": None}


class InvestorInputsTests(unittest.TestCase):
    def test_scenario_calculations_use_assumptions_and_preserve_negative_cash_flow(self):
        row = {"fiscal_year": 2025, "revenue": 100, "revenue_growth_pct": 10, "net_margin_pct": 20,
               "operating_cash_flow_margin_pct": 30, "capex_to_revenue_pct": 40}
        inputs = default_assumptions(row)
        results = project_scenarios(row, inputs, {"available": True, "quote": {"market_cap": 200}})
        middle = results[1]
        self.assertAlmostEqual(middle["revenue"], 110)
        self.assertAlmostEqual(middle["net_income"], 22)
        self.assertAlmostEqual(middle["free_cash_flow"], -11)
        self.assertAlmostEqual(middle["market_cap_to_scenario_net_income"], 200/22)
        self.assertEqual(row["revenue"], 100)
        inputs[1]["net_margin_pct"] = -10
        self.assertIsNone(project_scenarios(row, inputs, {"available": True, "quote": {"market_cap": 200}})[1]["market_cap_to_scenario_net_income"])

    def test_missing_and_invalid_assumptions_do_not_become_zero_or_infinity(self):
        row = {"fiscal_year": 2025, "revenue": 100}
        inputs = default_assumptions(row)
        self.assertIsNone(project_scenarios(row, inputs)[1]["revenue"])
        for bad in (float("nan"), float("inf"), -101, True):
            inputs = default_assumptions(row)
            inputs[0]["growth_pct"] = bad
            with self.subTest(bad=bad), self.assertRaises(SecDataError):
                project_scenarios(row, inputs)

    def test_quote_dates_old_years_and_nonpositive_profit_limit_valuation(self):
        report = extract("AMZN")
        row = report["rows"][-1]
        quote = quote_fixture()
        validate_quote(quote, get_company("AMZN"))
        context = valuation_context(report, row, quote, today=date(2026, 11, 19))
        self.assertTrue(context["available"])
        self.assertEqual(context["market_cap_to_annual_net_income"], 2000/row["net_income"])
        self.assertFalse(valuation_context(report, row, quote, today=date(2026, 12, 1))["available"])
        self.assertFalse(valuation_context(report, report["rows"][0], quote, today=date(2026, 11, 19))["available"])
        self.assertFalse(valuation_context(report, row, quote, today=date(2026, 11, 17))["available"])
        self.assertIsNone(valuation_context(report, {**row, "net_income": -1}, quote, today=date(2026,11,19))["market_cap_to_annual_net_income"])
        with self.assertRaises(SecDataError):
            validate_quote({**quote, "market_cap": 999}, get_company("AMZN"))

    def test_failed_quote_does_not_replace_the_previous_snapshot(self):
        with TemporaryDirectory() as directory:
            company = SimpleNamespace(ticker="AMZN", data_dir=Path(directory))
            path = Path(directory)/"valuation.json"
            path.write_text("old data")
            ticker = SimpleNamespace(get_info=lambda: {"symbol": "AMZN", "currency": "EUR"})
            with self.assertRaises(SecDataError):
                fetch_valuation(company, ticker_factory=lambda _: ticker)
            self.assertEqual(path.read_text(), "old data")

    def test_saved_assumptions_belong_to_their_input_snapshot(self):
        row = {"fiscal_year": 2025, "revenue_growth_pct": 10}
        original = default_assumptions(row)
        changed = [dict(item) for item in original]
        changed[1]["growth_pct"] = 25
        with TemporaryDirectory() as directory:
            company = SimpleNamespace(data_dir=Path(directory))
            save_assumptions(company, {"raw_sha256": "first"}, row, changed)
            self.assertEqual(load_assumptions(company, {"raw_sha256": "first"}, row)[1]["growth_pct"], 25)
            self.assertEqual(load_assumptions(company, {"raw_sha256": "updated"}, row)[1]["growth_pct"], 10)

    def test_educational_examples_distinguish_percentage_points_and_negative_fcf(self):
        lessons = metric_lessons({"revenue_growth_pct": 10, "operating_margin_pct": 22, "fcf_margin_pct": -5, "current_ratio": None}, {"operating_margin_pct": 20})
        self.assertIn("110,0", lessons["growth"]["example"])
        self.assertIn("2,0 prosenttiyksikköä", lessons["profitability"]["change"])
        self.assertIn("negatiivinen", lessons["cash_flow"]["example"])
        self.assertIn("puuttuu", lessons["balance"]["example"])

    def test_price_status_and_scenario_evidence_cannot_be_invented(self):
        packet = brief_packet()
        analysis = brief_fixture()
        validate_brief(analysis, packet)
        analysis.investment_case.price_assessment = "available"
        with self.assertRaisesRegex(SecDataError, "hintatietojen"):
            validate_brief(analysis, packet)
        analysis = brief_fixture()
        analysis.outlook.evidence[0].source_id = "M2025_revenue"
        with self.assertRaisesRegex(SecDataError, "skenaario"):
            validate_brief(analysis, packet)

    def test_available_price_case_requires_both_price_and_scenario(self):
        packet = brief_packet()
        packet["valuation"]["available"] = True
        packet["sources"]["V_quote"] = {"text": "Dated quote", "url": "https://finance.yahoo.com/quote/AMZN/"}
        analysis = brief_fixture()
        analysis.investment_case.price_assessment = "available"
        from learning.company_analysis import Evidence
        analysis.investment_case.evidence = [Evidence(source_id="V_quote", quote=""), Evidence(source_id="S_continuity", quote="")]
        validate_brief(analysis, packet)
        analysis.investment_case.evidence.pop()
        with self.assertRaisesRegex(SecDataError, "Sijoitusarvio"):
            validate_brief(analysis, packet)

    def test_exact_quote_can_be_relocated_but_an_invented_quote_cannot(self):
        from learning.company_brief import canonicalize_evidence
        from learning.company_analysis import Evidence
        packet = brief_packet()
        packet['sources']['R0002'] = {'text': 'Actual source sentence about growth.', 'url': 'https://www.sec.gov/fixture.htm'}
        analysis = brief_fixture()
        analysis.topics[0].evidence.append(Evidence(source_id='R0001', quote='Actual source sentence about growth.'))
        repaired, changes = canonicalize_evidence(analysis, packet)
        self.assertEqual(repaired.topics[0].evidence[-1].source_id, 'R0002')
        self.assertEqual(analysis.topics[0].evidence[-1].source_id, 'R0001')
        self.assertTrue(changes)
        validate_brief(repaired, packet)
        analysis.topics[0].evidence[-1].quote = 'Invented quotation not in any excerpt.'
        repaired, changes = canonicalize_evidence(analysis, packet)
        with self.assertRaises(SecDataError):
            validate_brief(repaired, packet)
